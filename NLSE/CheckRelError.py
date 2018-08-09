from keras.callbacks import Callback 
import numpy as np
import timeit

class CheckRelError(Callback):
    def __init__(self, X_train, Y_train, X_test, Y_test,
                verbose = False, period=1):
        super(CheckRelError, self).__init__()
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.verbose = verbose
        self.period = period
        self.best_err_train     = 100
        self.best_err_test      = 100
        self.best_err_train_max = 100
        self.best_err_test_max  = 100
        self.best_err_train_ave = 100
        self.best_err_test_ave  = 100

    def on_train_begin(self, logs=None):
        self.best_err_train     = 100
        self.best_err_test      = 100
        self.best_err_train_max = 100
        self.best_err_test_max  = 100
        self.best_err_train_ave = 100
        self.best_err_test_ave  = 100

    def on_epoch_end(self, epoch, logs=None):
        
        if((epoch)%self.period == 0):
            t1 = timeit.default_timer()

            err_train = rel_error(self.model, self.X_train, self.Y_train)
            err_test  = rel_error(self.model, self.X_test, self.Y_test)

            betr = np.mean(err_train)
            bete = np.mean(err_test)
            betrm = np.amax(err_train)
            betem = np.amax(err_test)

            if(self.best_err_train+self.best_err_test > betr + bete):

                self.best_err_train = betr
                self.best_err_test = bete

            bestAv = (self.best_err_train_ave+self.best_err_test_ave+
                (self.best_err_train_max+self.best_err_test_max)/5)
            tempAv = (betr+bete+(betrm+betem)/5) 

            if ( bestAv > tempAv ):
                self.best_err_train_ave = betr
                self.best_err_train_max = betrm
                self.best_err_test_ave  = bete
                self.best_err_test_max  = betem

            t2 = timeit.default_timer()
            if(self.verbose):
                print("runtime of checkresult = %.2f secs" % (t2-t1))
                print('best train and test error = %.1e, %.1e' % 
                        (self.best_err_train, self.best_err_test))
                print('best train and test error ave/max = %.1e, %.1e, %.1e, %.1e' % 
                        (self.best_err_train_ave, self.best_err_train_max, 
                         self.best_err_test_ave, self.best_err_test_max))

        def on_train_end(self, logs=None):
            print('best train and test error = %.1e, %.1e' % (self.best_err_train, self.best_err_test))
            print('best train and test error ave/max = %.1e, %.1e, %.1e, %.1e' % (self.best_err_train_ave, self.best_err_train_max, self.best_err_test_ave, self.best_err_test_max))

def rel_error(model, X, Y):
    Yhat = model.predict(X)
    dY = Yhat - Y  
    axis = tuple(range(np.size(dY.shape))[1:])
    return  np.sqrt(np.sum(dY**2, axis = axis)/
                     np.sum(Y**2, axis = axis))

#We may want to save th emodel with the lowest relative error 
# class ModelRelErrCheckpoint(Callback):

#     def __init__(self, filepath, verbose=False,
#                  save_best_only=True, save_weights_only=True,
#                  mode='auto', period=1):
#         super(ModelCheckpoint, self).__init__()
#         self.verbose = verbose
#         self.filepath = filepath
#         self.save_best_only = save_best_only
#         self.save_weights_only = save_weights_only
#         self.period = period
#         self.epochs_since_last_save = 0


#     def on_epoch_end(self, epoch, logs=None):
#         logs = logs or {}
#         self.epochs_since_last_save += 1
#         if self.epochs_since_last_save >= self.period:
#             self.epochs_since_last_save = 0
#             filepath = self.filepath.format(epoch=epoch + 1, **logs)
#             if self.save_best_only:
#                 current = logs.get(self.monitor)
#                 if current is None:
#                     warnings.warn('Can save best model only with %s available, '
#                                   'skipping.' % (self.monitor), RuntimeWarning)
#                 else:
#                     if self.monitor_op(current, self.best):
#                         if self.verbose > 0:
#                             print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
#                                   ' saving model to %s'
#                                   % (epoch + 1, self.monitor, self.best,
#                                      current, filepath))
#                         self.best = current
#                         if self.save_weights_only:
#                             self.model.save_weights(filepath, overwrite=True)
#                         else:
#                             self.model.save(filepath, overwrite=True)
#                     else:
#                         if self.verbose > 0:
#                             print('\nEpoch %05d: %s did not improve from %0.5f' %
#                                   (epoch + 1, self.monitor, self.best))
#             else:
#                 if self.verbose > 0:
#                     print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
#                 if self.save_weights_only:
#                     self.model.save_weights(filepath, overwrite=True)
#                 else:
#                     self.model.save(filepath, overwrite=True)
