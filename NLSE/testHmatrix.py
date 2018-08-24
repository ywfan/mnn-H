# coding=utf-8
# vim: sw=4 et tw=100
""" code for MNN-H.
reference:
    Y Fan, L Lin, L Ying, L Zepeda-Nunez,
    A multiscale neural network based on hierarchical matrices,
    arXiv preprint arXiv:1807.01883

    written by Yuwei Fan (ywfan@stanford.edu)
"""
import os
import numpy as np
import h5py
# ------------------ keras ----------------
from keras.models import Model
from keras.layers import Input, Conv1D, Flatten, Lambda
from keras.layers import Add, Reshape

from keras import optimizers
from keras.utils import plot_model

from utils import period_padding_1d, init_parser, output_funs
from utils import CheckRelError, rel_error

# ===================== Setting Parameters ====================
# using a parser to obtain the argument, see utils.py for more details
parser = init_parser('NLSE - MNN-H', input_prefix='nlse2v2', L=6,
                     percent=0.5, train_res='trainresultH.txt')
args = parser.parse_args()

# setup: parameters
alpha = args.alpha
L     = args.L
N_cnn = args.n_cnn

# preparation for output
data_path = 'data/'
log_path = 'logs/'
if not os.path.exists(log_path):
    os.mkdir(log_path)

output_filename = log_path + 'tHL' + str(L) + 'Nc' + str(N_cnn) + 'Al' + str(alpha)
if args.output_suffix is None:
    output_filename += str(os.getpid()) + '.txt'
else:
    output_filename += args.output_suffix + '.txt'

log = open(output_filename, "w+")
# defining output functions (see utils.py for more details)
(output, outputnewline, outputvec) = output_funs(log)

# ==================== Loading Data ========================
filenameIpt = data_path + 'Input_'  + args.input_prefix + '.h5'
filenameOpt = data_path + 'Output_' + args.input_prefix + '.h5'

# import data: size of data: Nsamples * Nx
fInput = h5py.File(filenameIpt, 'r')
InputArray = fInput['Input'][:]
fOutput = h5py.File(filenameOpt, 'r')
OutputArray = fOutput['Output'][:]
(Nsamples, Nx) = InputArray.shape
assert OutputArray.shape == InputArray.shape

output(args)
outputnewline()
output('Input data filename     = %s' % filenameIpt)
output('Output data filename    = %s' % filenameOpt)
output("Nx                      = %d" % Nx)
output("Nsamples                = %d" % Nsamples)
outputnewline()

(n_input, n_output) = (Nx, Nx)

# train data
n_train = min(int(Nsamples * args.percent), 20000)
n_test  = max(n_train, min(Nsamples - n_train, 5000))

if args.batch_size is None:
    BATCH_SIZE = n_train // 100
else:
    BATCH_SIZE = args.batch_size

# ========= pre-treating and splitting the data =============
Nsamples = InputArray.shape[0]
mean_out = np.mean(OutputArray[0:n_train, :])
mean_in  = np.mean(InputArray[0:n_train, :])
output("mean of input / output is %.6f\t %.6f" % (mean_in, mean_out))
InputArray /= mean_in * 2
InputArray -= 0.5
OutputArray -= mean_out

X_train = InputArray[0:n_train, :]
Y_train = OutputArray[0:n_train, :]
X_test  = InputArray[(Nsamples-n_test):Nsamples, :]
Y_test  = OutputArray[(Nsamples-n_test):Nsamples, :]
output("[n_input, n_output] = [%d, %d]" % (X_train.shape[1], Y_train.shape[1]))
output("[n_train, n_test] = [%d, %d]" % (n_train, n_test))
# reshaping hte input data
X_train = np.reshape(X_train, X_train.shape + (1,))
X_test  = np.reshape(X_test,  X_test.shape  + (1,))


# parameters
# Nx = 2^L *m
m = Nx // (2**L)
output('m = %d' % m)

def error(model, X, Y):
    return rel_error(model, X, Y, meanY=mean_out)


# ==================== Building the Network ========================
(n_b_ad, n_b_2, n_b_l) = (1, 2, 3)  # see the paper arXiv:1807.01883
# u = \sum_{l=2}^L u_l + u_ad
# u_l = U M V^T v
Ipt = Input(shape=(n_input, 1))
u_list = []  # list of u_l and u_ad
for ll in range(2, L+1):
    w = m * 2**(L-ll)
    # restriction
    Vv = Conv1D(alpha, w, strides=w, activation='linear')(Ipt)
    # kernel
    MVv = Vv
    n_b = n_b_2 if ll == 2 else n_b_l
    for k in range(0, N_cnn):
        MVv = Lambda(lambda x: period_padding_1d(x, 2*n_b+1))(MVv)
        MVv = Conv1D(alpha, 2*n_b+1, activation='relu')(MVv)

    # interpolation
    u_l = Conv1D(w, 1, activation='linear')(MVv)
    u_l = Flatten()(u_l)
    u_list.append(u_l)

# Adjacent part
u_ad = Reshape((n_input//m, m))(Ipt)
for k in range(0, N_cnn):
    act_fun = 'linear' if k == (N_cnn-1) else 'relu'
    u_ad = Lambda(lambda x: period_padding_1d(x, 2*n_b_ad+1))(u_ad)
    u_ad = Conv1D(m, 2*n_b_ad+1, activation=act_fun)(u_ad)

u_ad = Flatten()(u_ad)
u_list.append(u_ad)

Opt = Add()(u_list)

# ==================== Training the Network ========================
model = Model(inputs=Ipt, outputs=Opt)
try:
    plot_model(model, to_file='mnnH.png', show_shapes=True)
except ImportError:
    print("plot_model is not supported on this device")

optimizers.Nadam(lr=args.lr, schedule_decay=0.004)
model.compile(loss='mean_squared_error', optimizer='Nadam')

output('number of params      = %d' % model.count_params())
outputnewline()
model.summary()

checkrelerror = CheckRelError(X_train=X_train, Y_train=Y_train,
                              X_test=X_test, Y_test=Y_test,
                              verbose=True, period=10, error_fun=error)
model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=args.epoch,
          verbose=args.verbose, callbacks=[checkrelerror])

log.close()

log_os = open(args.sum_file, "a")
log_os.write('%s\t%d\t%d\t%d\t' % (args.input_prefix, alpha, L, N_cnn))
log_os.write('%d\t%d\t%d\t%d\t' % (BATCH_SIZE, n_train, n_test, model.count_params()))
log_os.write('%.3e\t%.3e\t' % (checkrelerror.best_err_train, checkrelerror.best_err_test))
log_os.write('%.3e\t%.3e\t' % (checkrelerror.best_err_train_max, checkrelerror.best_err_test_max))
log_os.write('\n')
log_os.close()
