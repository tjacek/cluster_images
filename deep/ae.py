import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import lasagne,pickle
import numpy as np
import theano
import theano.tensor as T
import utils.files as files
import deep.tools as tools
import deep,convnet
#from basic.external import make_imgs 

class Autoencoder(object):
    def __init__(self,hyperparams,in_var,l_hid,l_out,
                     reduction,reconstruction,loss,updates):
        self.hyperparams=hyperparams
        self.in_var=in_var
        self.l_hid=l_hid
        self.l_out=l_out
    	self.prediction=theano.function([self.in_var], reduction,allow_input_downcast=True)
        self.reconstructed=theano.function([self.in_var], 
                                           reconstruction,allow_input_downcast=True)
        self.loss=theano.function([in_var], loss,allow_input_downcast=True,name="Train")
        self.updates=theano.function([in_var], loss, 
                               updates=updates,allow_input_downcast=True)

    def get_model(self):
        data = lasagne.layers.get_all_param_values(self.l_out)
        return deep.Model(self.hyperparams,data)
    
    def set_model(self,model):
        lasagne.layers.set_all_param_values(self.l_out,model.params)

    def get_numpy(self):
        return [self.l_hid.W.get_value(),self.l_hid.b.get_value()]

    def __str__(self):
        return str(self.hyperparams)

def default_parametrs():
    return {"num_input":(None,7200),"num_hidden":600,"batch_size":100}	

def build_autoencoder(hyper_params=None):
    if(hyper_params==None):
        hyper_params=default_parametrs()
    num_hidden=hyper_params["num_hidden"]
    input_shape = hyper_params["num_input"]
    l_in =  lasagne.layers.InputLayer(shape=input_shape)
    l_hid = lasagne.layers.DenseLayer(l_in, num_units=num_hidden)
    l_out = lasagne.layers.DenseLayer(l_hid, num_units=input_shape[1])
    reconstruction = lasagne.layers.get_output(l_out)
    reduction=lasagne.layers.get_output(l_hid)
    in_var=l_in.input_var
    # target_var = T.ivector('targets')
    loss=get_loss(reconstruction,in_var)
    updates=get_updates(loss,l_out)
    return   Autoencoder(hyper_params,in_var,l_hid,l_out,
                         reduction,reconstruction,loss,updates)

def get_loss(reconstruction,in_var):    
    loss = lasagne.objectives.squared_error(reconstruction, in_var)
    loss = loss.mean()
    #l_hid=all_layers["out"]
    #l1_penalty = regularize_layer_params(l_hid, l1) * 0.001
    return loss #+ l1_penalty    

def get_updates(loss,out_layer):
    params = lasagne.layers.get_all_params(out_layer, trainable=True)
    return lasagne.updates.nesterov_momentum(
                 loss, params, learning_rate=0.1, momentum=0.8) 

def read_ae(path):
    with open(path, 'r') as f:
        model = pickle.load(f)
    #nn.layers.set_all_param_values(model, data)
    conv_net=build_autoencoder(model.hyperparams)
    conv_net.set_model(model)
    return conv_net

if __name__ == "__main__": 
    path_dir="../dataset0a/cats"
    #imgset=np.array(make_imgs(path_dir,norm=True))
    #model=build_autoencoder(default_parametrs())
    #model=deep.test_unsuper_model(imgset,model)
    #model.get_model().save("../dataset0a/ae")