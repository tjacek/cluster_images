import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

def get_number_of_batches(dataset,batch_size):
    n_batches=len(dataset)/batch_size
    if((len(dataset) % batch_size) != 0):
        n_batches+=1 
    return n_batches

def get_batches(dataset,n_batches,batch_size):
    data=dataset.get_data()
    indexes= range(n_batches)
    batchs=[get_batch(i,batch_size,data) for i in indexes]
    return batchs

def get_batch(i,batch_size,data):
    return data[i:(i+1)*batch_size]

def standarized_images(dataset):
    return get_batches(dataset,len(dataset),1)

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

def get_sigmoid(x,w,b):
    return T.nnet.sigmoid(T.dot(x,w) + b)

def get_crossentropy_loss(x,y,z):
    L = - T.sum(x * T.log(z) + (1 - x) * T.log(1 - z), axis=1)
    return T.mean(L)

def comput_updates(loss, params, learning_rate=0.05):
    gparams = [T.grad(loss, param) for param in params]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(params, gparams)
    ]
    return (loss,updates)
