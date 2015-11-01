import numpy as np
import theano

def init_zeros(size):
    return np.zeros(size,dtype=theano.config.floatX)

def init_random(height,width,rng):
    n_units=width + height
    lower_bound=-4 * np.sqrt(6. / n_units)
    upper_bound=4 * np.sqrt(6. / n_units)
    dim=(width, height)
    raw_data=rng.uniform(low=lower_bound,high=upper_bound,size=dim)
    return np.asarray(raw_data,dtype=theano.config.floatX)
