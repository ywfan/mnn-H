# Multiscale Neural Network (MNN) based on hierarchical matrices
This repo is the code for the paper: https://arxiv.org/abs/1807.01883

The code to generate data is written by __MATLAB__ and the neural network is implemented by __python__ based on _keras_ on top of _tensorflow_

## How to run
1. generate data by matlab code
2. run the code in the tensorflow environment 

For NLSE, the main file of the matlab code is _NLSEsample.m_ for 1d and _NLSEsample2d.m_ for 2d.

The neural network use _argparse_ to set the parameters. One can use 
```bash
	python testHmatrix.py --help  
```
to print the usage of all the parameters and its default values.
One example to run the code:
```bash
	python testHmatrix.py --epoch 2000 --alpha 4 --output-suffix V1
```


The code for Kohn-Sham map is same as that for NLSE, thus we only provide the code for NLSE.
