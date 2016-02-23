import numpy as np
import theano
import theano.tensor as T
import deep

def show_dim(layer):
    print("input")
    print(layer.input_shape)
    print("output")
    print(layer.output_shape)

def get_batch(imgs,batch_size=10):
    n_batches=get_n_batches(imgs,batch_size)
    batches=[imgs[i*batch_size:(i+1)*batch_size] for i in range(n_batches)]
    return [np.array(batch_i) for batch_i in batches],n_batches

def get_n_batches(img,batch_size=10):
    return (len(img)/batch_size)+1
