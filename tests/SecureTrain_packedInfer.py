## type"taskset -c 0 python3 serial_code.py ==> benchmark the time in a single core
## type"python3 serial_code.py" ==> benchmark the time in serial HE order without specify the number of cores

import math
from seal import *
from seal_helper import *
import os
import numpy as np
from random import *
from mlxtend.data import loadlocal_mnist
import time


#Helper function: load the source data.
def loadMnist():
    data_train, label_train = loadlocal_mnist(
        images_path=os.getcwd()+'/train-images-idx3-ubyte',
        labels_path=os.getcwd()+'/train-labels-idx1-ubyte');
    data_test, label_test = loadlocal_mnist(
        images_path=os.getcwd()+'/t10k-images-idx3-ubyte',
        labels_path=os.getcwd()+'/t10k-labels-idx1-ubyte');
    #normalize the data
    data_train_norm = data_train.astype(np.float)/255.;
    data_test_norm = data_test.astype(np.float)/255.;
    return data_train_norm, data_test_norm, label_train, label_test

#Helper function: Get the relu result.
def ReLu(input):
    y = input.copy()
    np.maximum(y,0.,y);
    return DoubleVector(y)

#Helper function: generate random set from server (g1, g2, and v)
def g1_g2(hidden_dim):
    #generate the random numbers, v shouldn't be zero
    v = np.random.uniform(low=-10., high=10., size=(hidden_dim,))
    u = 1./v
    g1 = u.copy()
    g1[g1>0] = 0.
    g2 = np.abs(u)
    return DoubleVector(g1), DoubleVector(g2), DoubleVector(v)

def secure_train_inference_serial():
    #COMMON crypto parameters
    print_example_banner("Benchmark: SecureTrain Inference in Serial")
    parms = EncryptionParameters(scheme_type.CKKS)
    poly_modulus_degree = 8192
    parms.set_poly_modulus_degree(poly_modulus_degree)
    parms.set_coeff_modulus(CoeffModulus.Create(
        poly_modulus_degree, [60, 40, 40, 60]))
    scale = pow(2.0, 40)

    #BOTH sides use this context
    context = SEALContext.Create(parms)
    print_parameters(context)
    #BOTH party will have common encode and evaluate setting
    evaluator = Evaluator(context)
    encoder = CKKSEncoder(context)
    slot_count = encoder.slot_count()
    print("Number of slots: " + str(slot_count))

    #client will generate its own public key and secret key
    keygen = KeyGenerator(context)
    public_key = keygen.public_key()
    secret_key = keygen.secret_key()
    #client will have different encryption and decryption parameters
    encryptor = Encryptor(context, public_key)
    decryptor = Decryptor(context, secret_key)

    #server will generate its own public key and secret key
    keygen1 = KeyGenerator(context)
    public_key1 = keygen1.public_key()
    secret_key1 = keygen1.secret_key()
    #server will have different encryption and decryption parameters
    encryptor1 = Encryptor(context, public_key1)
    decryptor1 = Decryptor(context, secret_key1)

    ####for Layer one
    ######################client side######################
    #client read image from mnist data set
    x_train, x_test, y_train, y_test = loadMnist();
    #the hidden+output layer dims
    Hiddendim = [128,128,10]
    print("The number of layers is: ",len(Hiddendim)+1)
    #index of current layer
    CurrentLayer = 0
    #calculate the number of data that can be packed in one ciphertext
    num_one_cipher = slot_count//np.array(Hiddendim).max()
    start_time = time.time()
    #get num_one_cipher pieces of input data 
    x = x_train[0:num_one_cipher]
    #the random set generated at client
    x1 = np.random.uniform(low=-1., high=1., size=(num_one_cipher,x.shape[1]))
    #each data has a corresponding r1
    r1 = np.random.uniform(low=-1., high=1., size=(num_one_cipher,1))
    #each data has a corresponding r2
    r2 = np.random.uniform(low=-1., high=1., size=(num_one_cipher,1))
    w1_c = np.random.uniform(low=-1., high=1., size=(x.shape[1],Hiddendim[0]))
    b_c = np.random.uniform(low=-1., high=1., size=(1,Hiddendim[0]))
    #calculate based on the generated random set
    x2 = x - x1
    r1x1_Array = x1*r1
    r2x2_Array = x2*r2
    r2_xw1_c_b_c_Array = ((x.dot(w1_c)) + b_c) * r2
    #encrypt the data r2_div_r1
    r2_div_r1_plain = Plaintext()
    r2_div_r1_encrypted = Ciphertext()
    encoder.encode(DoubleVector(np.repeat(r2/r1,Hiddendim[0])), scale, r2_div_r1_plain)
    encryptor.encrypt(r2_div_r1_plain, r2_div_r1_encrypted)
    #encrypt the data r2_xw1_c_b_c
    r2_xw1_c_b_c_plain = Plaintext()
    r2_xw1_c_b_c_encrypted = Ciphertext()
    encoder.encode(DoubleVector(r2_xw1_c_b_c_Array.flatten()), scale, r2_xw1_c_b_c_plain)
    encryptor.encrypt(r2_xw1_c_b_c_plain, r2_xw1_c_b_c_encrypted)
    #encrypt the data r2
    r2_plain = Plaintext()
    r2_encrypted = Ciphertext()
    encoder.encode(DoubleVector(np.repeat(r2,Hiddendim[0])), scale, r2_plain)
    encryptor.encrypt(r2_plain, r2_encrypted)
    ############################################

    ######################data exchange######################

    ######################server side######################
    w1_s = np.random.uniform(low=-1., high=1., size=(x.shape[1],Hiddendim[0]))
    b1_s = np.random.uniform(low=-1., high=1., size=(1,Hiddendim[0]))
    #dot product
    r2_x2_ws = r2x2_Array.dot(w1_s)
    r1_x1_ws = r1x1_Array.dot(w1_s)
    #encode vectors for addition and multiplications
    r2_x2_ws_plain = Plaintext()
    r1_x1_ws_plain = Plaintext()
    b1_s_plain = Plaintext()
    encoder.encode(DoubleVector(r2_x2_ws.flatten()), scale, r2_x2_ws_plain)
    encoder.encode(DoubleVector(r1_x1_ws.flatten()), scale, r1_x1_ws_plain)
    encoder.encode(DoubleVector(np.repeat(b1_s,num_one_cipher,axis=0).flatten()), scale, b1_s_plain)
    #generate the pair of polar indicator
    g1, g2, v= g1_g2(Hiddendim[0]*num_one_cipher)
    #encode g1, g2, and v
    g1_plain = Plaintext()
    g2_plain = Plaintext()
    v_plain = Plaintext()
    encoder.encode(g1, scale, g1_plain)
    encoder.encode(g2, scale, g2_plain)
    encoder.encode(v, scale, v_plain)
    #Encrypt g1 and g2
    g1_encrypted = Ciphertext()
    g2_encrypted = Ciphertext()
    encryptor1.encrypt(g1_plain, g1_encrypted)
    encryptor1.encrypt(g2_plain, g2_encrypted)
    #begin data addition
    #get the r2_b1_s
    evaluator.multiply_plain_inplace(r2_encrypted, b1_s_plain)
    #get the r1_x1_w_s_r2_div_r1
    evaluator.multiply_plain_inplace(r2_div_r1_encrypted, r1_x1_ws_plain)
    #get the first addition at scale 40
    evaluator.add_plain_inplace(r2_xw1_c_b_c_encrypted, r2_x2_ws_plain)
    #rescale to 80
    plainTextOne = Plaintext()
    encoder.encode(1., scale, plainTextOne)
    evaluator.multiply_plain_inplace(r2_xw1_c_b_c_encrypted, plainTextOne)    
    #get the second addition at scale 80
    evaluator.add_inplace(r2_encrypted, r2_div_r1_encrypted)
    #get the final addition
    evaluator.add_inplace(r2_encrypted,r2_xw1_c_b_c_encrypted)
    #randomize the addition at scale 120
    evaluator.multiply_plain_inplace(r2_encrypted, v_plain)
    #THE FINAL RESULT IS in r2_encrypted
    ############################################

    ######################data exchange######################

    ######################client side######################
    #decrypt the ciphertext
    r2z_vs_plain = Plaintext()
    decryptor.decrypt(r2_encrypted, r2z_vs_plain)
    r2z_vs = DoubleVector()
    encoder.decode(r2z_vs_plain, r2z_vs)
    #resize the decrypted data and remove the random number
    y = (np.array(r2z_vs)[0:(Hiddendim[0]*num_one_cipher)].reshape((num_one_cipher,Hiddendim[0]))/r2).flatten()   
    #encode the y and relu(y)
    y_plain = Plaintext()
    relu_y_plain = Plaintext()
    encoder.encode(ReLu(y), scale, relu_y_plain)
    encoder.encode(DoubleVector(y), scale, y_plain)
    #get the relu result
    evaluator.multiply_plain_inplace(g1_encrypted, y_plain)
    evaluator.multiply_plain_inplace(g2_encrypted, relu_y_plain)
    evaluator.add_inplace(g1_encrypted, g2_encrypted)
    #THE RELU IS g1_encrypted    
    ####repeat from the second layer
    while CurrentLayer < (len(Hiddendim)-1):
        #generate the random set
        H_x1 = np.random.uniform(low=-5., high=5., size=(num_one_cipher,Hiddendim[CurrentLayer]))
        H_r1 = np.random.uniform(low=-5., high=5., size=(num_one_cipher,1))
        H_r2 = np.random.uniform(low=-5., high=5., size=(num_one_cipher,1))
        H_h1 = np.random.uniform()
        H_h2 = np.random.uniform()
        H_w1_c = np.random.uniform(low=-5., high=5., size=(Hiddendim[CurrentLayer],Hiddendim[CurrentLayer+1]))
        H_w_c = np.random.uniform(low=-5., high=5., size=(Hiddendim[CurrentLayer],Hiddendim[CurrentLayer+1]))
        H_b_c = np.random.uniform(low=-5., high=5., size=(1,Hiddendim[CurrentLayer+1]))
        #calculate the corresponding items
        #get w2_c
        H_w2_c = H_w_c - H_w1_c
        H_h1_w1_c = H_w1_c*H_h1
        H_h2_w2_c = H_w2_c*H_h2
        H_r2_x1w_c_b_c_Array = ((H_x1.dot(H_w_c)) + H_b_c) * H_r2
        H_r1x1_Array = H_x1*H_r1
        # encrypt the data
        H_r2_plain = Plaintext()
        H_r2_plain4x2 = Plaintext()
        H_r2_div_r1_plain = Plaintext()
        H_r2_x1w_c_b_c_plain = Plaintext()
        H_h1_div_plain = Plaintext()
        H_h2_div_plain = Plaintext()
        H_x1_plain = Plaintext()
        encoder.encode(DoubleVector(np.repeat(H_r2,Hiddendim[CurrentLayer+1])), scale, H_r2_plain)
        encoder.encode(DoubleVector(np.repeat(H_r2,Hiddendim[CurrentLayer])), scale, H_r2_plain4x2)
        encoder.encode(DoubleVector(np.repeat(H_r2/H_r1,Hiddendim[CurrentLayer+1])), scale, H_r2_div_r1_plain)
        encoder.encode(DoubleVector(H_r2_x1w_c_b_c_Array.flatten()), scale, H_r2_x1w_c_b_c_plain)
        encoder.encode(1./H_h1, scale, H_h1_div_plain)
        encoder.encode(1./H_h2, scale, H_h2_div_plain)
        H_r2_encrypted = Ciphertext()
        H_r2_div_r1_encrypted = Ciphertext()
        H_r2_x1w_c_b_c_encrypted = Ciphertext()
        H_h1_div_encrypted = Ciphertext()
        H_h2_div_encrypted = Ciphertext()        
        encryptor.encrypt(H_r2_plain, H_r2_encrypted)
        encryptor.encrypt(H_r2_div_r1_plain, H_r2_div_r1_encrypted)
        encryptor.encrypt(H_r2_x1w_c_b_c_plain, H_r2_x1w_c_b_c_encrypted)
        encryptor.encrypt(H_h1_div_plain, H_h1_div_encrypted)
        encryptor.encrypt(H_h2_div_plain, H_h2_div_encrypted)
        #Generate r2x2 from encrypted relu
        encoder.encode(DoubleVector(H_x1.flatten()), g1_encrypted.scale(), H_x1_plain)
        evaluator.sub_plain_inplace(g1_encrypted, H_x1_plain)
        evaluator.multiply_plain_inplace(g1_encrypted, H_r2_plain4x2)
        #THE RESULT r2x2 IS g1_encrypted
        ############################################

        ######################data exchange######################

        ######################server side######################
        #Generate random numbers
        H_w_s = np.random.uniform(low=-5., high=5., size=(Hiddendim[CurrentLayer],Hiddendim[CurrentLayer+1]))
        H_b_s = np.random.uniform(low=-5., high=5., size=(1,Hiddendim[CurrentLayer+1]))
        #generate g1 and g2 and v numbers
        H_g1, H_g2, H_v= g1_g2(Hiddendim[CurrentLayer+1]*num_one_cipher)
        #encode g1, g2, and v
        H_g1_plain = Plaintext()
        H_g2_plain = Plaintext()
        H_v_plain = Plaintext()
        encoder.encode(H_g1, scale, H_g1_plain)
        encoder.encode(H_g2, scale, H_g2_plain)
        encoder.encode(H_v, scale, H_v_plain)
        #Encrypt g1 and g2
        H_g1_encrypted = Ciphertext()
        H_g2_encrypted = Ciphertext()
        encryptor1.encrypt(H_g1_plain, H_g1_encrypted)
        encryptor1.encrypt(H_g2_plain, H_g2_encrypted)
        #decrypt the ciphertext
        H_r2_x2_plain = Plaintext()
        decryptor1.decrypt(g1_encrypted, H_r2_x2_plain)
        H_r2_x2_vec = DoubleVector()
        encoder.decode(H_r2_x2_plain, H_r2_x2_vec)
        H_r2_x2 = (np.array(H_r2_x2_vec)[0:(Hiddendim[CurrentLayer]*num_one_cipher)]).reshape((num_one_cipher,Hiddendim[CurrentLayer]))
        #do the following calculation after encryption
        H_r2_x2_ws = H_r2_x2.dot(H_w_s)
        H_r2_x2_h1_w1_c = H_r2_x2.dot(H_h1_w1_c)
        H_r2_x2_h2_w2_c = H_r2_x2.dot(H_h2_w2_c)
        H_r1x1_ws = H_r1x1_Array.dot(H_w_s)
        H_r2_x2_ws_plain = Plaintext()
        H_r2_x2_h1_w1_c_plain = Plaintext()
        H_r2_x2_h2_w2_c_plain = Plaintext()
        H_r1x1_ws_plain = Plaintext()
        H_b_s_plain = Plaintext()        
        encoder.encode(DoubleVector(H_r2_x2_ws.flatten()), scale, H_r2_x2_ws_plain)
        encoder.encode(DoubleVector(H_r2_x2_h1_w1_c.flatten()), scale, H_r2_x2_h1_w1_c_plain)
        encoder.encode(DoubleVector(H_r2_x2_h2_w2_c.flatten()), scale, H_r2_x2_h2_w2_c_plain)
        encoder.encode(DoubleVector(H_r1x1_ws.flatten()), scale, H_r1x1_ws_plain)
        encoder.encode(DoubleVector(np.repeat(H_b_s,num_one_cipher,axis=0).flatten()), scale, H_b_s_plain)
        #get Ciphertext
        evaluator.multiply_plain_inplace(H_r2_encrypted, H_b_s_plain)
        evaluator.multiply_plain_inplace(H_r2_div_r1_encrypted, H_r1x1_ws_plain)
        evaluator.multiply_plain_inplace(H_h2_div_encrypted, H_r2_x2_h2_w2_c_plain)
        evaluator.multiply_plain_inplace(H_h1_div_encrypted, H_r2_x2_h1_w1_c_plain)
        evaluator.add_inplace(H_r2_encrypted, H_r2_div_r1_encrypted)
        evaluator.add_inplace(H_r2_encrypted, H_h2_div_encrypted)
        evaluator.add_inplace(H_r2_encrypted, H_h1_div_encrypted)
        evaluator.add_plain_inplace(H_r2_x1w_c_b_c_encrypted, H_r2_x2_ws_plain)
        #rescale the H_r2_x1w_c_b_c_encrypted
        evaluator.multiply_plain_inplace(H_r2_x1w_c_b_c_encrypted, plainTextOne)
        evaluator.add_inplace(H_r2_encrypted, H_r2_x1w_c_b_c_encrypted)
        evaluator.multiply_plain_inplace(H_r2_encrypted, H_v_plain)
        #THE RESULT IS H_r2_encrypted
        ############################################

        ######################data exchange######################

        ######################client side######################        
        #decrypt the ciphertext
        H_r2z_vs_plain = Plaintext()
        decryptor.decrypt(H_r2_encrypted, H_r2z_vs_plain)
        H_r2z_vs = DoubleVector()
        encoder.decode(H_r2z_vs_plain, H_r2z_vs)
        H_y = ((np.array(H_r2z_vs)[0:(Hiddendim[CurrentLayer+1]*num_one_cipher)]).reshape(num_one_cipher,Hiddendim[CurrentLayer+1])/H_r2).flatten()
        #encode the y and relu(y)
        H_y_plain = Plaintext()
        H_relu_y_plain = Plaintext()
        encoder.encode(DoubleVector(H_y), scale, H_y_plain)
        encoder.encode(ReLu(H_y), scale, H_relu_y_plain)
        evaluator.multiply_plain_inplace(H_g1_encrypted, H_y_plain)
        evaluator.multiply_plain_inplace(H_g2_encrypted, H_relu_y_plain)
        evaluator.add_inplace(H_g1_encrypted, H_g2_encrypted)
        g1_encrypted = H_g1_encrypted
        #increase layer countered
        CurrentLayer = CurrentLayer + 1    
    print("The total time for inference in serial per input is (s): ", (time.time() - start_time)/num_one_cipher)


if __name__ == '__main__':

    #run the inference of SecureTrain
    secure_train_inference_serial()    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
