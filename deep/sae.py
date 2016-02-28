import lasagne
import numpy as np
import theano
import theano.tensor as T
import utils.files as files
import deep.tools as tools

class StackedAE(object):
    def __init__(self,autoencoder,n_cats):
    	W_hid,b_hid,W_out,b_out=autoencoder.get_numpy()
    	input_size,hidden_size=W_hid.shape
    	self.l_in =  lasagne.layers.InputLayer(shape=(None,input_size))
        self.l_hid = lasagne.layers.DenseLayer(self.l_in, num_units=hidden_size)
        softmax = lasagne.nonlinearities.softmax
        self.network = lasagne.layers.DenseLayer(self.l_hid,n_cats, nonlinearity=softmax)

    def get_loss(self):
        prediction = lasagne.layers.get_output(network)
        loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
        self.loss = loss.mean()

    def get_params(self):
        return lasagne.layers.get_all_params(self.l_out, trainable=True)
