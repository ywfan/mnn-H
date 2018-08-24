# coding=utf-8
# vim: sw=4 et tw=100
"""
  File with the utility functions
  Y Fan, L Lin, L Ying, L Zepeda-Nunez, A multiscale neural network based on hierarchical matrices,
  arXiv preprint arXiv:1807.01883
"""
import argparse
import timeit
import numpy as np
from keras import backend as K
from keras.callbacks import Callback

# functions
def period_padding_1d(x, size):
    """period padding for 3-tensor with shape: (batch_size, Nx, features)

    # Argument
        x: 3-tensor
        size: int
    """
    s = size // 2
    return K.concatenate([x[:, x.shape[1]-s:x.shape[1], :], x, x[:, 0:(size-s-1), :]], axis=1)


# channels last, i.e. x.shape = [batch_size, nx, ny, n_channels]
def period_padding_2d(x, size_x, size_y):
    """period padding for 4-tensor with shape: (batch_size, Nx, Ny, features)

    # Argument
        x: 4-tensor
        size_x, size_y: int
    """
    wx = size_x // 2
    wy = size_y // 2
    nx = x.shape[1]
    ny = x.shape[2]
    # x direction
    y = K.concatenate([x[:, nx-wx:nx, :, :], x, x[:, 0:wx, :, :]], axis=1)
    # y direction
    z = K.concatenate([y[:, :, ny-wy:ny, :], y, y[:, :, 0:wy, :]], axis=2)
    return z


def matrix2tensor(x, wx, wy):
    """ Reshape a tensor to matrix by blocks

    # Arguments
        wx, wy: int

    # Input shape: 4D tensor with shape (batch_size, Nx, Ny, features)
    # Output shape: 4D tensor with shape (batch_size, Nx*wx, Ny*wy, features//(wx, wy))
    """
    nx = int(x.shape[1])
    ny = int(x.shape[2])
    nw = int(x.shape[3])
    assert nw == 1
    assert nx % wx == 0
    assert ny % wy == 0
    y = K.reshape(x, (-1, nx//wx, wx, ny//wy, wy))
    z = K.permute_dimensions(y, (0, 1, 3, 2, 4))
    return K.reshape(z, (-1, nx//wx, ny//wy, wx*wy))


def tensor2matrix(x, wx, wy):
    """ Reshape a matrix to tensor by blocks, inverse of matrix2tensor

    # Arguments
        wx, wy: int

    # Input shape: 4D tensor with shape (batch_size, Nx, Ny, features)

    # Output shape: 4D tensor with shape (batch_size, Nx//wx, Ny//wy, features*wx*wy)
    """
    nx = int(x.shape[1])
    ny = int(x.shape[2])
    w2 = int(x.shape[3])
    assert w2 == wx * wy
    y = K.reshape(x, (-1, nx, ny, wx, wy))
    z = K.permute_dimensions(y, (0, 1, 3, 2, 4))
    return K.reshape(z, (-1, nx*wx, ny*wy))


def output_funs(log):
    """define the output """
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

    return (output, outputnewline, outputvec)


def init_parser(descStr, epoch=4000, input_prefix='nlse2v2', alpha=6, L=6,
                n_cnn=5, lr=0.001, batch_size=None,
                verbose=2, output_suffix=None, percent=0.5,
                train_res='trainresultH.txt'):
    """set the input parameters"""
    parser = argparse.ArgumentParser(description=descStr)
    parser.add_argument('--epoch', type=int, default=epoch, metavar='N',
                        help='input number of epochs for training (default: %(default)s)')
    parser.add_argument('--input-prefix', type=str, default=input_prefix, metavar='N',
                        help='prefix of input data filename (default: %(default)s)')
    parser.add_argument('--alpha', type=int, default=alpha, metavar='N',
                        help='number of channels for training (default: %(default)s)')
    parser.add_argument('--L', type=int, default=L, metavar='N',
                        help='number of grids (L, Nx=2^L*mx) (default: %(default)s)')
    parser.add_argument('--n-cnn', type=int, default=n_cnn, metavar='N',
                        help='number layer of CNNs (default: %(default)s)')
    parser.add_argument('--lr', type=float, default=lr, metavar='LR',
                        help='learning rate (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=batch_size, metavar='N',
                        help='batch size (default: #train samples/100)')
    parser.add_argument('--verbose', type=int, default=verbose, metavar='N',
                        help='verbose (default: %(default)s)')
    parser.add_argument('--output-suffix', type=str, default=output_suffix, metavar='N',
                        help='suffix output filename(default: )')
    parser.add_argument('--percent', type=float, default=percent, metavar='precent',
                        help='percentage of number of total data(default: %(default)s)')
    parser.add_argument('--sum-file', type=str, default=train_res, metavar='N',
                        help='file to summarize the error of the training (default: %(default)s)')
    return parser


def rel_error(model, X, Y, meanY=0):
    """function to compute the relative error
    model : trained neural network to compute the predicted data
    X     : input data
    Y     : reference output data
    meanY : the mean of the of the untreated Y """
    Yhat = model.predict(X)
    axis = tuple(range(1, X.ndim-1))
    return np.linalg.norm(Y - Yhat, axis=axis) / np.linalg.norm(Y+meanY, axis=axis)


class CheckRelError(Callback):
    """check the best result"""
    def __init__(self, X_train, Y_train, X_test, Y_test,
                 verbose=False, period=1, error_fun=rel_error):
        super(CheckRelError, self).__init__()
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.verbose = verbose
        self.period = period
        self.error_fun = error_fun  # function to compute the error
        self.best_err_train     = 1
        self.best_err_test      = 1
        self.best_err_train_max = 1
        self.best_err_test_max  = 1
        self.best_err_train_ave = 1
        self.best_err_test_ave  = 1

    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1) % self.period == 0:
            t1 = timeit.default_timer()
            err_train = self.error_fun(self.model, self.X_train, self.Y_train)
            err_test  = self.error_fun(self.model, self.X_test, self.Y_test)
            betr = np.mean(err_train)
            bete = np.mean(err_test)
            betrm = np.amax(err_train)
            betem = np.amax(err_test)

            if(self.best_err_train+self.best_err_test > betr + bete):
                self.best_err_train = betr
                self.best_err_test = bete

            bestAv = (self.best_err_train_ave + self.best_err_test_ave +
                      (self.best_err_train_max+self.best_err_test_max)/5)
            tempAv = (betr+bete+(betrm+betem)/5)

            if bestAv > tempAv:
                self.best_err_train_ave = betr
                self.best_err_train_max = betrm
                self.best_err_test_ave  = bete
                self.best_err_test_max  = betem

            t2 = timeit.default_timer()
            if self.verbose:
                print("runtime of checkresult = %.2f secs" % (t2-t1))
                print('best train and test error = %.1e, %.1e' %
                      (self.best_err_train, self.best_err_test))
                print('best train and test error ave/max = %.1e, %.1e, %.1e, %.1e' %
                      (self.best_err_train_ave, self.best_err_train_max,
                       self.best_err_test_ave, self.best_err_test_max))

    def on_train_end(self, logs=None):
        print('best train and test error = %.1e, %.1e' % (self.best_err_train,
                                                          self.best_err_test))
        print('best train and test error ave/max = \
              %.1e, %.1e, %.1e, %.1e' % (self.best_err_train_ave,
                                         self.best_err_train_max,
                                         self.best_err_test_ave,
                                         self.best_err_test_max))
