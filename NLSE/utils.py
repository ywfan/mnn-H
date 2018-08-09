"""
  File with the utility functions 
  Y Fan, L Lin, L Ying, L Zepeda-Nunez, A multiscale neural network based on hierarchical matrices,
  arXiv preprint arXiv:1807.01883

  
"""
import numpy as np
from keras import backend as K
import argparse

# functions
def padding(x, size):
    return K.concatenate([x[:,x.shape[1]-size//2:x.shape[1],:], x, x[:,0:(size-size//2-1),:]], axis=1)

def rel_err_loss(y_true, y_pred):
    return K.sum(K.square(y_pred - y_true), axis=-1)/K.sum(K.square(y_true), axis=-1)

def initParser():
	parser = argparse.ArgumentParser(description='NLSE - MNN-H')
	parser.add_argument('--epoch', type=int, default=4000, metavar='N',
	                    help='input number of epochs for training (default: %(default)s)')
	parser.add_argument('--input-prefix', type=str, default='nlse2v2', metavar='N',
	                    help='prefix of input data filename (default: %(default)s)')
	parser.add_argument('--alpha', type=int, default=6, metavar='N',
	                    help='number of channels for training (default: %(default)s)')
	parser.add_argument('--L', type=int, default=7, metavar='N',
	                    help='number of grids (L+1, N=2^L*m) (default: %(default)s)')
	parser.add_argument('--n-cnn', type=int, default=5, metavar='N',
	                    help='number layer of CNNs (default: %(default)s)')
	parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
	                    help='learning rate (default: %(default)s)')
	parser.add_argument('--batch-size', type=int, default=0, metavar='N',
	                    help='batch size (default: #train samples/100)')
	parser.add_argument('--verbose', type=int, default=2, metavar='N',
	                    help='verbose (default: %(default)s)')
	parser.add_argument('--output-suffix', type=str, default='None', metavar='N',
	                    help='suffix output filename(default: )')
	parser.add_argument('--percent', type=float, default=0.5, metavar='precent',
	                    help='percentage of number of total data(default: %(default)s)')
	parser.add_argument('--sum-file', type=str, default='trainresultH.txt', metavar='N',
	                    help='file to summarize the error of the training (default: %(default)s)')
	return parser

def preTreatDataNSLE(InputArray, OutputArray,n_train,n_test,output):
	"""preTreatData(InputArray, OutputArray,n_train,n_test)
	function to pre-treat the data for the NSLE
	"""
	# obtaining the number of Samples 
	Nsamples = InputArray.shape[0]
	# computing mean of the input and output data
	mean_out = np.mean(OutputArray[0:n_train, :])
	mean_in  = np.mean(InputArray[0:n_train, :])

	output("mean of input / output is %.6f\t %.6f" % (mean_in, mean_out))
	
	# treating the data
	InputArray /= mean_in * 2
	InputArray -= 0.5
	OutputArray -= mean_out
	
	# splitting the data 
	X_train = InputArray[0:n_train, :] 
	Y_train = OutputArray[0:n_train, :]
	X_test  = InputArray[(Nsamples-n_test):Nsamples, :]
	Y_test  = OutputArray[(Nsamples-n_test):Nsamples, :]
	n_input = X_train.shape[1]
	n_output= Y_train.shape[1]
	
	output("[n_input, n_output] = [%d, %d]" % (n_input, n_output))
	output("[n_train, n_test] = [%d, %d]" % (n_train, n_test))

	# reshaping hte input data 
	X_train = np.reshape(X_train, [X_train.shape[0], X_train.shape[1], 1])
	X_test  = np.reshape(X_test,  [X_test.shape[0],  X_test.shape[1],  1])

	return (X_train,Y_train,X_test,Y_test)