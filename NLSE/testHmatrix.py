"""
  code for MNN-H.
  reference:
  Y Fan, L Lin, L Ying, L Zepeda-Núnez, A multiscale neural network based on hierarchical matrices,
  arXiv preprint arXiv:1807.01883

  written by Yuwei Fan (ywfan@stanford.edu)
"""
# ------------------ keras ----------------
from keras.models import Sequential, Model
# layers
from keras.layers import Input, Conv1D, LocallyConnected1D, Flatten, Lambda
from keras.layers import Add, multiply, dot, Reshape

from keras import backend as K
from keras import regularizers, optimizers
from keras.engine.topology import Layer
from keras.constraints import non_neg
from keras.utils import np_utils
#from keras.utils import plot_model
from keras.callbacks import LambdaCallback, ReduceLROnPlateau

K.set_floatx('float32')

import os
import timeit
import argparse
import h5py
import numpy as np
import random
import math

parser = argparse.ArgumentParser(description='NLSE - MNN-H')
parser.add_argument('--epoch', type=int, default=4000, metavar='N',
                    help='input number of epochs for training (default: 4000)')
parser.add_argument('--input-prefix', type=str, default='nlse2v2', metavar='N',
                    help='prefix of input data filename (default: nlse2v2)')
parser.add_argument('--alpha', type=int, default=6, metavar='N',
                    help='number of channels for training (default: 6)')
parser.add_argument('--k-grid', type=int, default=7, metavar='N',
                    help='number of grids (L+1, N=2^L*m) (default: 7)')
parser.add_argument('--n-cnn', type=int, default=5, metavar='N',
                    help='number layer of CNNs (default: 5)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--batch-size', type=int, default=0, metavar='N',
                    help='batch size (default: #samples/100)')
parser.add_argument('--verbose', type=int, default=2, metavar='N',
                    help='verbose (default: 2)')
parser.add_argument('--output-suffix', type=str, default='None', metavar='N',
                    help='suffix output filename(default: )')
parser.add_argument('--percent', type=float, default=0.5, metavar='precent',
                    help='percentage of number of total data(default: 0.5)')
args = parser.parse_args()
# setup: parameters
N_epochs = args.epoch
alpha = args.alpha
k_multigrid = args.k_grid
N_cnn = args.n_cnn
lr = args.lr

best_err_train = 1e-2
best_err_test = 1e-2
best_err_T_train = 10
best_err_T_test = 10
best_err_train_max = 10
best_err_test_max = 10

# preparation for output
data_path = 'data/'
log_path = 'logs/'
if not os.path.exists(log_path):
  os.mkdir(log_path)
outputfilename = log_path + 'tHL' + str(k_multigrid) + 'Nc' + str(N_cnn);
#outputmodel = log_path + 'tHL' + str(k_multigrid) + 'Nc' + str(N_cnn);
if(args.output_suffix == 'None'):
    outputfilename += str(os.getpid()) + '.txt'
    #outputmodel += str(os.getpid()) + '.h5'
else:
    outputfilename += args.output_suffix + '.txt'
    #outputmodel += args.output_suffix + '.h5'
os = open(outputfilename, "w+")

def output(obj):
    print(obj)
    os.write(str(obj)+'\n')
def outputnewline():
    os.write('\n')
    os.flush()

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

# pre-treat the data
mean_v = 1
InputArray /= 40
InputArray -= 0.5
OutputArray -= mean_v

n_input = Nx
n_output = Nx

# train data
n_train = int(Nsamples * args.percent)
n_train = min(n_train, 20000)
n_test = Nsamples - n_train
n_test = min(n_train, n_test)
n_test = max(n_test, 5000)

if args.batch_size == 0:
    BATCH_SIZE = n_train // 100
else:
    BATCH_SIZE = args.batch_size

X_train = InputArray[0:n_train, :] #equal to 0:(n_train-1) in matlab
Y_train = OutputArray[0:n_train, :]
X_test  = InputArray[(Nsamples-n_test):Nsamples, :]
Y_test  = OutputArray[(Nsamples-n_test):Nsamples, :]

output("[n_input, n_output] = [%d, %d]" % (n_input, n_output))
output("[n_train, n_test] = [%d, %d]" % (n_train, n_test))

X_train = np.reshape(X_train, [X_train.shape[0], X_train.shape[1], 1])
X_test  = np.reshape(X_test,  [X_test.shape[0],  X_test.shape[1], 1])

# parameters
# Nx = 2^L *m, L = k_multigrid-1
m = Nx // (2**(k_multigrid - 1))
output('m = %d' % m)

# functions
def padding(x, size):
    return K.concatenate([x[:,x.shape[1]-size//2:x.shape[1],:], x, x[:,0:(size-size//2-1),:]], axis=1)

# test
def test_data(X, Y, string):
    Yhat = model.predict(X)
    dY = Yhat - Y
    errs = np.linalg.norm(dY, axis=1) / np.linalg.norm(Y+mean_v, axis=1)
    output("max/ave error of %s data:\t %.1e %.1e" % (string, np.amax(errs), np.mean(errs)))
    return errs

flag = True
def checkresult(epoch, step):
    global best_err_train, best_err_test, best_err_train_max, best_err_test_max, flag, best_err_T_train, best_err_T_test
    t1 = timeit.default_timer()
    if((epoch+1)%step == 0):
        err_train = test_data(X_train, Y_train, 'train')
        err_test  = test_data(X_test, Y_test, 'test')
        if(best_err_train > np.mean(err_train)):
            best_err_train = np.mean(err_train)
            best_err_test = np.mean(err_test)
            best_err_train_max = np.amax(err_train)
            best_err_test_max = np.amax(err_test)
            best_err_T_train = np.var(err_train)
            best_err_T_test = np.var(err_test)
            #IA = np.reshape(InputArray, [InputArray.shape[0], InputArray.shape[1], 1])
            #Yhat = model.predict(IA)
            #hf = h5py.File(outputmodel, 'w')
            #hf.create_dataset('prediction', data=Yhat)
        t2 = timeit.default_timer()
        if(flag):
          output("runtime of checkresult = %.2f secs" % (t2-t1))
          flag = False
        output('best train and test error = %.1e, %.1e,\t fit time    = %.1f secs' % (best_err_train, best_err_test, (t2 - start)))
        output('best train and test error var, max = %.1e, %.1e, %.1e, %.1e' % (best_err_T_train, best_err_train_max, best_err_T_test, best_err_test_max))
        outputnewline()

def outputvec(vec, string):
    os.write(string+'\n')
    for i in range(0, vec.shape[0]):
        os.write("%.6e\n" % vec[i])

n_b_ad = 1 # see the paper arXiv:1807.01883
n_b_2 = 2
n_b_l = 3
# u = \sum_{l=2}^L u_l + u_ad
# u_l = U M V^T v
Ipt = Input(shape=(n_input, 1))
u_list = [] # list of u_l and u_ad
for k in range(2, k_multigrid):
    w = m * 2**(k_multigrid-k-1)
    #restriction: we can add more layers
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

# adjcent part
u_ad = Reshape((n_input//m, m))(Ipt)
for i in range(0, N_cnn-1):
  u_ad = Lambda(lambda x: padding(x, 2*n_b_ad+1))(u_ad)
  u_ad = Conv1D(m, 2*n_b_ad+1, activation='relu')(u_ad)

u_ad = Lambda(lambda x: padding(x, 2*n_b_ad+1))(u_ad)
u_ad = Conv1D(m, 2*n_b_ad+1, activation='linear')(u_ad)
u_ad = Flatten()(u_ad)
u_list.append(u_ad)

Opt = Add()(u_list)

# model
model = Model(inputs=Ipt, outputs=Opt)
#plot_model(model, to_file='mnnH.png', show_shapes=True)
model.compile(loss='mean_squared_error', optimizer='Nadam')
model.optimizer.schedule_decay = (0.004)
output('number of params      = %d' % model.count_params())
outputnewline()
model.summary()

start = timeit.default_timer()
RelativeErrorCallback = LambdaCallback(
        on_epoch_end=lambda epoch, logs: checkresult(epoch, 10))
#ReduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10,
#        verbose=1, mode='auto', cooldown=0, min_lr=1e-6)
model.optimizer.lr = (lr)
model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=N_epochs, verbose=args.verbose,
        callbacks=[RelativeErrorCallback])

checkresult(1,1)
err_train = test_data(X_train, Y_train, 'train')
err_test  = test_data(X_test, Y_test, 'test')
outputvec(err_train, 'Error for train data')
outputvec(err_test,  'Error for test data')

os.close()

log_os = open('trainresultH.txt', "a")
log_os.write('%s\t%d\t%d\t%d\t' % (args.input_prefix, alpha, k_multigrid, N_cnn))
log_os.write('%d\t%d\t%d\t%d\t' % (BATCH_SIZE, n_train, n_test, model.count_params()))
log_os.write('%.3e\t%.3e\t' % (best_err_train, best_err_test))
log_os.write('%.3e\t%.3e\t%.3e\t%.3e\t' % (best_err_train_max, best_err_test_max, best_err_T_train, best_err_T_test))
log_os.write('\n')
log_os.close()
