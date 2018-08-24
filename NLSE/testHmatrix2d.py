# coding=utf-8
# vim: sw=4 et tw=100
"""
  code for MNN-H 2d.
  reference:
  Y Fan, L Lin, L Ying, L Zepeda-Núnez, A multiscale neural network based on hierarchical matrices,
  arXiv preprint arXiv:1807.01883

  written by Yuwei Fan (ywfan@stanford.edu)
"""
import os
import numpy as np
import h5py
# ------------------ keras ----------------
from keras.models import Model
from keras.layers import Input, Conv2D
from keras.layers import Add, Lambda

from keras import optimizers
# from keras.utils import plot_model

from utils import initParser, outputFunc, period_padding2D, matrix2tensor, tensor2matrix
from utils import CheckRelError, rel_error

# ===================== Setting Parameters ====================
parser = initParser('NLSE - MNN-H 2d', input_prefix='nlse2d2', L=4,
                    percent=2./3., trainResStr='trainresult2dH.txt')
args = parser.parse_args()

# setup: parameters
alpha = args.alpha
L     = args.L
N_cnn = args.n_cnn

Nsamples = 300

# preparation for output
data_path = 'data2d/'
log_path = 'logs2d/'
if not os.path.exists(log_path):
    os.mkdir(log_path)

output_filename = log_path + 't2dHL' + str(L) + 'Nc' + str(N_cnn) + 'Al' + str(alpha)
if args.output_suffix is None:
    output_filename += str(os.getpid()) + '.txt'
else:
    output_filename += args.output_suffix + '.txt'

log = open(output_filename, "w+")
# defining output functions (see utils.py for more details)
(output, outputnewline, outputvec) = outputFunc(log)

# ==================== Loading Data ========================
filenameIpt = data_path + 'Input_'  + args.input_prefix + '.h5'
filenameOpt = data_path + 'Output_' + args.input_prefix + '.h5'

print('Reading data...')
fInput = h5py.File(filenameIpt, 'r')
InputArray = fInput['Input'][:, :, 0:Nsamples]
fOutput = h5py.File(filenameOpt, 'r')
OutputArray = fOutput['Output'][:, :, 0:Nsamples]
print('Reading data finished')

InputArray = np.transpose(InputArray, (2, 1, 0))
OutputArray = np.transpose(OutputArray, (2, 1, 0))
print(InputArray.shape)

assert InputArray.shape[0] == Nsamples
_, Nx, Ny = InputArray.shape
assert OutputArray.shape == (Nsamples, Nx, Ny)

output(args)
outputnewline()
output('Input data filename     = %s' % filenameIpt)
output('Output data filename    = %s' % filenameOpt)
output("(Nx, Ny)                = (%d, %d)" % (Nx, Ny))
output("Nsamples                = %d" % Nsamples)
outputnewline()

n_input = InputArray.shape[1:3]
n_output = OutputArray.shape[1:3]

# train data
n_train = min(int(Nsamples * args.percent), 30000)
n_test  = max(n_train, min(Nsamples - n_train, 5000))

if args.batch_size is None:
    BATCH_SIZE = n_train // 100
else:
    BATCH_SIZE = args.batch_size

# pre-treat the data
mean_out = np.mean(OutputArray[0:n_train, :, :])
mean_in  = np.mean(InputArray[0:n_train, :, :])
output("mean of input / output is %.6f\t %.6f" % (mean_in, mean_out))
InputArray /= mean_in * 2
InputArray -= 0.5
OutputArray -= mean_out

X_train = InputArray[0:n_train, :, :]
Y_train = OutputArray[0:n_train, :, :]
X_test  = InputArray[n_train:(n_train+n_test), :, :]
Y_test  = OutputArray[n_train:(n_train+n_test), :, :]

output("[n_input, n_output] = [(%d,%d),  (%d,%d)]" % (n_input + n_output))
output("[n_train, n_test]   = [%d, %d]" % (n_train, n_test))

X_train = np.reshape(X_train, X_train.shape + (1,))
X_test  = np.reshape(X_test,  X_test.shape  + (1,))

# parameters
m = (Nx // (2**L), Ny // (2**L))
output('m = (%d, %d)' % m)

def error(model, X, Y):
    return rel_error(model, X, Y, meanY=mean_out)


# ==================== Building the Network ========================
(w_b_ad, w_b_2, w_b_l) = ((3, 3), (5, 5), (7, 7))
Ipt = Input(shape=n_input+(1,))
u_list = []
for ll in range(2, L+1):
    w = tuple(n * 2**(L-ll) for n in m)
    # restriction
    Vv = Conv2D(alpha, w, strides=w, activation='linear')(Ipt)
    # kernel
    MVv = Vv
    w_b = w_b_2 if ll == 2 else w_b_l
    for k in range(0, N_cnn):
        MVv = Lambda(lambda x: period_padding2D(x, w_b[0], w_b[1]))(MVv)
        MVv = Conv2D(alpha, w_b, activation='relu')(MVv)

    # interpolation
    u_l = Conv2D(w[0]*w[1], (1, 1), activation='linear')(MVv)
    u_l = Lambda(lambda x: tensor2matrix(x, w[0], w[1]))(u_l)
    u_list.append(u_l)

# adjacent
u_ad = Lambda(lambda x: matrix2tensor(x, w[0], w[1]))(Ipt)
for k in range(0, N_cnn):
    act_fun = 'linear' if k == (N_cnn-1) else 'relu'
    u_ad = Lambda(lambda x: period_padding2D(x, w_b_ad[0], w_b_ad[1]))(u_ad)
    u_ad = Conv2D(m[0]*m[1], w_b_ad, activation=act_fun)(u_ad)

u_ad = Lambda(lambda x: tensor2matrix(x, m[0], m[1]))(u_ad)
u_list.append(u_ad)

Opt = Add()(u_list)

# ==================== Training the Network ========================
model = Model(inputs=Ipt, outputs=Opt)
model.compile(loss='mean_squared_error', optimizer='Nadam')
model.optimizer.schedule_decay = (0.004)
output('number of params      = %d' % model.count_params())
outputnewline()
model.summary()

checkrelerror = CheckRelError(X_train=X_train, Y_train=Y_train,
                              X_test=X_test, Y_test=Y_test,
                              verbose=True, period=10, errorFun=error)
model.optimizer.lr = (args.lr)
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
