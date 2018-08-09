"""
  code for MNN-H.
  reference:
  Y Fan, L Lin, L Ying, L Zepeda-Nunez, A multiscale neural network based on hierarchical matrices,
  arXiv preprint arXiv:1807.01883

  written by Yuwei Fan (ywfan@stanford.edu)
"""
# ------------------ keras ----------------
from keras.models import Model
# layers
from keras.layers import Input, Conv1D, Flatten, Lambda
from keras.layers import Add, Reshape

from keras import backend as K
from keras import regularizers, optimizers
from keras.engine.topology import Layer
from CheckRelError import CheckRelError, rel_error
from utils import padding, initParser, rel_err_loss, splitData
#from keras.utils import plot_model
from keras.callbacks import LambdaCallback

import os
import timeit
import argparse
import h5py
import numpy as np
import random
import math

##################### Setting Parameters ########################

K.set_floatx('float32')
# using a parser to obtain the argument
# see utils.py for more details
parser   = initParser('NLSE - MNN-H')
args     = parser.parse_args()
# setup: parameters
N_epochs = args.epoch
alpha    = args.alpha
L        = args.L
N_cnn    = args.n_cnn
lr       = args.lr
sum_file = args.sum_file
percent  = args.percent

# preparation for output
data_path = 'data/'
log_path = 'logs/'
if not os.path.exists(log_path):
    os.mkdir(log_path)
outputfilename = log_path + 'tHL' + str(L) + 'Nc' + str(N_cnn)

if(args.output_suffix == 'None'):
    outputfilename += str(os.getpid()) + '.txt'
else:
    outputfilename += args.output_suffix + '.txt'

# LZ: shall we change the name of the file?
log = open(outputfilename, "w+")

# drfining output functions
def output(obj):
    print(obj)
    log.write(str(obj)+'\n')
def outputnewline():
    log.write('\n')
    log.flush()
def outputvec(vec, string):
    log.write(string+'\n')
    for i in range(0, vec.shape[0]):
        log.write("%.6e\n" % vec[i])

##################### Loading Data ########################

filenameIpt = data_path + 'Input_'  + args.input_prefix + '.h5'
filenameOpt = data_path + 'Output_' + args.input_prefix + '.h5'

# import data: size of data: Nsamples * Nx
fInput = h5py.File(filenameIpt,'r')
InputArray = fInput['Input'][:]

fOutput = h5py.File(filenameOpt,'r')
OutputArray = fOutput['Output'][:]

Nsamples = InputArray.shape[0]
Nx = InputArray.shape[1]

output(args)
outputnewline()
output('Input data filename     = %s' % filenameIpt)
output('Output data filename    = %s' % filenameOpt)
output("Nx                      = %d" % Nx)
output("Nsamples                = %d" % Nsamples)
outputnewline()

assert OutputArray.shape[0] == Nsamples
assert OutputArray.shape[1] == Nx

n_input  = Nx
n_output = Nx

# train data
n_train = min(int(Nsamples * args.percent), 20000)
n_test  = Nsamples - n_train
n_test  = max(min(n_train, n_test), 5000)

#choosing the batch size
if args.batch_size == 0:
    batch_size = n_train // 100
else:
    batch_size = args.batch_size

########## pre-treating and splitting the data #############

Nsamples = InputArray.shape[0]
# computing mean of the input and output data
mean_out = np.mean(OutputArray[0:n_train, :])
mean_in  = np.mean(InputArray[0:n_train, :])

output("mean of input / output is %.6f\t %.6f" % (mean_in, mean_out))

# treating the data
InputArray /= mean_in * 2
InputArray -= 0.5
OutputArray -= mean_out

(X_train,Y_train,X_test,Y_test) = splitData(InputArray,
                                            OutputArray,
                                            n_train,
                                            n_test,
                                            output)
# parameters
# Nx = 2^L *m, L = L-1
m = Nx // (2**(L - 1))
output('m = %d' % m)

def error(model, X, Y): return  rel_error(model, X, Y,
                                          meanY = mean_out)

##################### Building the Network ########################
n_b_ad = 1 # see the paper arXiv:1807.01883
n_b_2 = 2
n_b_l = 3
# u = \sum_{l=2}^L u_l + u_ad
# u_l = U M V^T v
Ipt = Input(shape=(n_input, 1))
u_list = [] # list of u_l and u_ad
for k in range(2, L):
    w = m * 2**(L-k-1)
    #restriction
    Vv = Conv1D(alpha, w, strides=w, activation='linear')(Ipt)
    #kernel
    MVv = Vv
    if(k==2):
        for i in range(0,N_cnn):
          MVv = Lambda(lambda x: padding(x, 2*n_b_2+1))(MVv)
          MVv = Conv1D(alpha, 2*n_b_2+1, activation='relu')(MVv)
    else:
        for i in range(0,N_cnn):
          MVv = Lambda(lambda x: padding(x, 2*n_b_l+1))(MVv)
          MVv = Conv1D(alpha, 2*n_b_l+1, activation='relu')(MVv)
    #interpolation
    u_l = Conv1D(w, 1, activation='linear')(MVv)
    u_l = Flatten()(u_l)
    u_list.append(u_l)

# Adjcent part
u_ad = Reshape((n_input//m, m))(Ipt)
for i in range(0, N_cnn-1):
  u_ad = Lambda(lambda x: padding(x, 2*n_b_ad+1))(u_ad)
  u_ad = Conv1D(m, 2*n_b_ad+1, activation='relu')(u_ad)

u_ad = Lambda(lambda x: padding(x, 2*n_b_ad+1))(u_ad)
u_ad = Conv1D(m, 2*n_b_ad+1, activation='linear')(u_ad)
u_ad = Flatten()(u_ad)
u_list.append(u_ad)

Opt = Add()(u_list)

##################### Training the Network ########################
model = Model(inputs=Ipt, outputs=Opt)
#plot_model(model, to_file='mnnH.png', show_shapes=True)
optimizers.Nadam(lr=lr, schedule_decay=0.004)
model.compile(loss=rel_err_loss, optimizer='Nadam')

output('number of params      = %d' % model.count_params())
outputnewline()
model.summary()

start = timeit.default_timer()
checkrelerror= CheckRelError(X_train = X_train, Y_train = Y_train,
                             X_test  = X_test,  Y_test  = Y_test,
                             verbose = True,    period  = 10,
                             errorFun = error)

model.fit(X_train, Y_train,
          batch_size = batch_size,   epochs    = N_epochs,
          verbose    = args.verbose, callbacks = [checkrelerror])

err_train = error(model, X_train, Y_train)
err_test  = error(model, X_test, Y_test)

# I don't know if really necessary
# outputvec(err_train, 'Error for train data')
# outputvec(err_test,  'Error for test data')

log.close()

log_os = open(sum_file, "a")
log_os.write('%s\t%d\t%d\t%d\t' % (args.input_prefix, alpha,\
                                   L, N_cnn))
log_os.write('%d\t%d\t%d\t%d\t' % (batch_size, n_train, \
                                   n_test, model.count_params()))
log_os.write('%.3e\t%.3e\t' % (checkrelerror.best_err_train, \
                               checkrelerror.best_err_test))
log_os.write('%.3e\t%.3e\t' % (checkrelerror.best_err_train_max, \
                               checkrelerror.best_err_test_max))
                                           # best_err_T_train, best_err_T_test))
log_os.write('\n')
log_os.close()
