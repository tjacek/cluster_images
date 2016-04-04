import numpy as np
import theano
import theano.tensor as T
import lasagne

class SimpleLayer(object):
    def __init__(self,W,b):
        self.W=W
        self.b=b

    def get_params(self):
        return [self.W, self.b] 

def make_simple_layer(in_size,out_size,postfix="f"):
    ort_init=lasagne.init.Orthogonal()
    cons_init=lasagne.init.Constant(0.)
    W_value=ort_init.sample((in_size,out_size))
    b_value=cons_init.sample((out_size,))
    W=make_var(W_value,"W_"+postfix)
    b=make_var(b_value,"b_"+postfix)
    return SimpleLayer(W,b)

def make_var(value,name):
    return theano.shared(value=value,name=name,borrow=True)

def dist_to_category(dist):
    return dist.flatten().argmax(axis=0)

def show_dim(layer):
    print("input")
    print(layer.input_shape)
    print("output")
    print(layer.output_shape)

def to_dist(index,n_cats):
    dist=np.zeros((1,n_cats),dtype='int64')
    dist[index]=1
    return dist	

def get_batch(imgs,batch_size=10):
    n_batches=get_n_batches(imgs,batch_size)
    batches=[imgs[i*batch_size:(i+1)*batch_size] for i in range(n_batches)]
    return [np.array(batch_i) for batch_i in batches],n_batches

def get_n_batches(img,batch_size=10):
    return (len(img)/batch_size)+1