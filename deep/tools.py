import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np
import theano
import theano.tensor as T
import lasagne
import deep
import ae

class SimpleLayer(object):
    def __init__(self,W,b):
        self.W=W
        self.b=b

    def get_params(self):
        return [self.W, self.b] 

class SigmoidLayer(SimpleLayer):
    def symb_expr(self,x):
        return T.nnet.sigmoid(T.dot(x,self.W) + self.b)

class TanhLayer(SimpleLayer):
    def symb_expr(self,x):
        return T.nnet.tanh(T.dot(x,self.W) + self.b)        

def make_simple_layer(in_size,out_size,postfix="f",layer=TanhLayer):
    ort_init=lasagne.init.Orthogonal()
    cons_init=lasagne.init.Constant(0.)
    W_value=ort_init.sample((in_size,out_size))
    b_value=cons_init.sample((out_size,))
    W=make_var(W_value,"W_"+postfix)
    b=make_var(b_value,"b_"+postfix)
    return layer(W,b)

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

def reconstruction(path):
    nn=ae.read_ae(path)

if __name__ == "__main__": 
    ae_path="../dataset0a/ae"
    reconstruction(ae_path)