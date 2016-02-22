import lasagne
import numpy as np
import theano
import theano.tensor as T

class AutoencoderModel(object):
    def __init__(self,W,b,b_prime):
        self.W=W
        self.b=b
        self.b_prime=b_prime
        self.W_prime = W.T

    def get_params(self):
        return [self.W, self.b, self.b_prime]

class Autoencoder(object):
    def __init__(self,hyper_params):
    	num_hidden=hyper_params["num_hidden"]
    	input_shape = (1,hyper_params["num_input"])
        self.l_in =  lasagne.layers.InputLayer(shape=input_shape)#,input_var=self.input_var)
        print(self.l_in.output_shape)
        self.l_hid = lasagne.layers.DenseLayer(self.l_in, num_units=num_hidden)
        show_dim(self.l_hid)
        self.l_rec = lasagne.layers.DenseLayer(self.l_hid, num_units=input_shape[1])
        show_dim(self.l_rec)        
        self.l_out = self.l_rec#
        #self.l_out =lasagne.layers.InverseLayer(self.l_rec,self.l_hid)  #self.l_in)
        #print(self.l_out.output_shape()) 
        self.prediction = lasagne.layers.get_output(self.l_out)
        self.get_loss()

    def get_input_var(self):
        return self.l_in.input_var

    def get_loss(self):
    	target_var=self.get_input_var()
    	self.loss = lasagne.objectives.squared_error(self.prediction,target_var)
        self.loss = self.loss.mean()

    def get_params(self):
        return lasagne.layers.get_all_params(self.l_out, trainable=True)

    def get_vars(self):
    	return self.l_hid.W,self.l_hid.b

    def get_updates(self):
    	params=self.get_params()
        return lasagne.updates.nesterov_momentum(
        	     self.loss, params, learning_rate=0.01, momentum=0.9)	

    def get_numpy(self):
    	W,b=self.get_vars()
        return W.get_value(),b.get_value()

def default_parametrs():
    return {"num_input":3600,"num_hidden":600}	

def train_model(imgs,hyper_params,num_iter=50):
    model=Autoencoder(hyper_params)
    input_var=model.get_input_var()
    updates=model.get_updates()
    train_fn = theano.function([input_var], model.loss, updates=updates)
    #pred = theano.function([input_var], model.prediction)
    input_dim=(1,hyper_params["num_input"])
    for epoch in range(num_iter):
        #for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
        for img_i in imgs:
            img_i=img_i.reshape(input_dim) #.flatten()
            print(img_i.shape) 
            #inputs, targets = batch
            loss_i=train_fn(img_i)
            #print(pred(img_i).shape)
            print(str(epoch) + " "+str(loss_i))
    return model

def show_dim(layer):
    print("input")
    print(layer.input_shape)
    print("output")
    print(layer.output_shape)