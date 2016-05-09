import lasagne
import numpy as np
import theano
import theano.tensor as T
import utils.files as files
import deep.tools as tools
import convnet

class Autoencoder(object):
    def __init__(self,hyper_params,in_var,target_var,
                     reduction,reconstruction,loss,updates):
        self.hyper_params=hyper_params
        self.in_var=in_var
    	self.prediction=theano.function([self.in_var], reduction,allow_input_downcast=True)
        self.reconstructed=theano.function([self.in_var], 
                                           reconstruction,allow_input_downcast=True)
        self.loss=theano.function([in_var,target_var], loss,allow_input_downcast=True)
        self.updates=theano.function([in_var, target_var], loss, 
                               updates=updates,allow_input_downcast=True)

    def get_model(self):
        data = lasagne.layers.get_all_param_values(self.l_out)
        return convnet.Model(self.hyperparams,data)
    
    def set_model(self,model):
        lasagne.layers.set_all_param_values(self.l_out,model.params)

def default_parametrs():
    return {"num_input":(None,3600),"num_hidden":600}	

def build_autoencoder(params):
    num_hidden=hyper_params["num_hidden"]
    input_shape = hyper_params["num_input"]
    l_in =  lasagne.layers.InputLayer(shape=input_shape)
    l_hid = lasagne.layers.DenseLayer(self.l_in, num_units=num_hidden)
    l_out = lasagne.layers.DenseLayer(self.l_hid, num_units=input_shape[1])
    reconstruction = lasagne.layers.get_output(l_out)
    reduction=lasagne.layers.get_output(l_hid)
    in_var=in_layer.input_var
    target_var = T.ivector('targets')
    loss=get_loss(reconstruction,in_var,target_var)
    updates=get_updates(loss,l_out)
    return   Autoencoder(hyper_params,in_var,target_var,
                         reduction,reconstruction,loss,updates)

def get_loss(reconstruction,in_var,target_var):    
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    #l_hid=all_layers["out"]
    #l1_penalty = regularize_layer_params(l_hid, l1) * 0.001
    return loss #+ l1_penalty    

def get_updates(loss,out_layer):
    params = lasagne.layers.get_all_params(out_layer, trainable=True)
    return lasagne.updates.nesterov_momentum(
                 self.loss, params, learning_rate=0.01, momentum=0.8)   