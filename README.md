## SecureTrain Implementation

This is a prototyping implementation of SecureTrain: An Efficient and Approximation-Free Framework for Privacy-Preserved Neural Network Training.

This implementation relies on: 

1. Microsoft [**SEAL**](https://github.com/microsoft/SEAL): an easy-to-use open-source ([MIT licensed](https://github.com/microsoft/SEAL/blob/master/LICENSE)) homomorphic encryption library developed by the Cryptography Research group at Microsoft.

2. [**pybind11**](https://github.com/pybind/pybind11): a lightweight header-only library that exposes C++ types in Python and vice versa, mainly to create Python bindings of existing C++ code.

We build SecureTrain by the [**python binding for the Microsoft SEAL library**](https://github.com/Huelse/SEAL-Python).

## Build with Linux
CMake (>= 3.10), GNU G++ (>= 6.0) or Clang++ (>= 5.0), Python (>=3.6.8)

`sudo apt-get update && sudo apt-get install g++ make cmake git python3 python3-dev python3-pip`

`git clone https://github.com/ChiaoThon/SecureTrain.git`

```shell
cd SEAL/native/src
cmake .
make
# return to the main directory
cd ../../..

pip3 install -r requirements.txt

# Setuptools (Recommend)
python3 setup.py build_ext -i
```

Docs: [setuptools](https://docs.python.org/3/distutils/configfile.html) [pybind11](https://pybind11.readthedocs.io/en/master/index.html)

Errors: If errors happen during the above process, first refer to the [**python binding for the Microsoft SEAL library**](https://github.com/Huelse/SEAL-Python)

## Tests

`cd tests`

run

```shell
#run with single core
taskset -c 0 python3 [example_name].py
```
 
or

```shell
#run with cores available
python3 [example_name].py
```

* The `.so` file must be in the same folder. In another words, you should put the `.so` file in main path to the `tests` file. 



## Getting Started
| Python files     | Description                                                  |
| ---------------- | ------------------------------------------------------------ |
| SecureTrain.py   | The overall inference performance of SecureTrain|
| SecureTrain_Inf_accuracy.py  | Test the inference accuracy of SecureTrain |
| SecureTrain_packedInfer.py  | Pack multiple inputs to enable parallel inference |



## Future
* generate share within lowest modulus
* make training part available
* make real-interaction part available


## About
This project is still testing now, if any problems(bugs), [Issue](https://github.com/ChiaoThon/SecureTrain/issues) please.


