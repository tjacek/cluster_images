import numpy as np
import theano
from theano.tensor.shared_randomstreams import RandomStreams

def init_zeros(size):
    return np.zeros(size,dtype=theano.config.floatX)

def init_random(height,width,rng):
    n_units=width + height
    lower_bound=-4 * np.sqrt(6. / n_units)
    upper_bound=4 * np.sqrt(6. / n_units)
    dim=(width, height)
    raw_data=rng.uniform(low=lower_bound,high=upper_bound,size=dim)
    return np.asarray(raw_data,dtype=theano.config.floatX)

def make_var(value,name):
    return theano.shared(value=value,name=name,borrow=True)

def make_rng(theano_rng=None):
    numpy_rng = np.random.RandomState(123)
    if not theano_rng:
        theano_rng=RandomStreams(numpy_rng.randint(2 ** 30))
    return numpy_rng,theano_rng

