import lasagne
import numpy as np
import theano
import theano.tensor as T
import utils.files as files
import deep.tools as tools

class AutoencoderModel(object):
    def __init__(self,W,b,W_prime,b_prime):
        self.W=W
        self.b=b
        self.b_prime=b_prime
        self.W_prime = W_prime

    def get_params(self):
        return [self.W, self.b, self.b_prime]

class Autoencoder(object):
    def __init__(self,hyper_params):
    	num_hidden=hyper_params["num_hidden"]
    	input_shape = (None,hyper_params["num_input"])
        self.l_in =  lasagne.layers.InputLayer(shape=input_shape)
        #print(self.l_in.output_shape)
        self.l_hid = lasagne.layers.DenseLayer(self.l_in, num_units=num_hidden)
        show_dim(self.l_hid)
        self.l_rec = lasagne.layers.DenseLayer(self.l_hid, num_units=input_shape[1])
        show_dim(self.l_rec)        
        self.l_out = self.l_rec#
        self.prediction_symb = lasagne.layers.get_output(self.l_out)
        self.get_loss()
        reduced=lasagne.layers.get_output(self.l_hid)
        self.prediction=theano.function([self.get_input_var()], reduced)

    def get_input_var(self):
        return self.l_in.input_var

    def get_loss(self):
    	target_var=self.get_input_var()
    	self.loss = lasagne.objectives.squared_error(self.prediction_symb,target_var)
        self.loss = self.loss.mean()

    def get_params(self):
        return lasagne.layers.get_all_params(self.l_out, trainable=True)

    def get_vars(self):
    	return [self.l_hid.W,self.l_hid.b,self.l_rec.W,self.l_rec.b]

    def get_updates(self):
    	params=self.get_params()
        return lasagne.updates.nesterov_momentum(
        	     self.loss, params, learning_rate=0.01, momentum=0.9)	

    def get_model(self):
        np_vars=self.get_numpy()
        return AutoencoderModel(np_vars[0],np_vars[1],np_vars[2],np_vars[3])

    def apply(self,img):
        img=img.reshape((1,3600)) #flatten()
        red_img=self.prediction(img)
        print(red_img.shape)
        return red_img.flatten()

    def get_numpy(self):
    	var=self.get_vars()
        return [ var_i.get_value() for var_i in var]

def default_parametrs():
    return {"num_input":3600,"num_hidden":600,"batch_size":100}	

def apply_autoencoder(imgs,ae_path):
    ae=files.read_object(ae_path)
    return [ae.apply(img_i) for img_i in imgs]
