import numpy as np
import theano

def init_zeros(size):
    return np.zeros(size,dtype=theano.config.floatX)
