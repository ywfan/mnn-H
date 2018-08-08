# Multiscale Neural Network (MNN) based on hierarchical matrices
This repo is the code for the paper: https://arxiv.org/abs/1807.01883
The generating code is written by MATLAB and the neural network is implemented by python based on keras on top of tensorflow

## How to run
- generate data by matlab code
- run the code in the tensorflow environment 

For NLSE, the main file of the matlab code is __NLSEsample.m__ for 1d and __NLSEsample2d.m__ for 2d.

The neural network use _argparse_ to set the parameters. One can use 
```
	python testHmatrix.py --help  
```
to print the usage of all the parameters and its default values.
