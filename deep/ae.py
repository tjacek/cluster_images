import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import lasagne,pickle
import numpy as np
import theano
import theano.tensor as T
import utils.files as files
import utils.imgs as imgs
import deep,convnet
from lasagne.regularization import regularize_layer_params, l2, l1
import deep.train as train

class Autoencoder(deep.NeuralNetwork):
    def __init__(self,hyperparams,out_layer,in_var,
                     reduction,reconstruction,loss,updates):
        super(Autoencoder,self).__init__(hyperparams,out_layer)
    	self.prediction=theano.function([in_var], reduction,allow_input_downcast=True)
        self.__reconstructed__=theano.function([in_var], 
                                           reconstruction,allow_input_downcast=True)
        self.loss=theano.function([in_var], loss,allow_input_downcast=True,name="loss")
        self.updates=theano.function([in_var], loss, 
                               updates=updates,allow_input_downcast=True)

    def reconstructed(self,in_img):
        print(type(in_img))
        org_dim=in_img.org_dim
        img2D=np.expand_dims(in_img,0)
        raw_rec=self.__reconstructed__(img2D)
        return imgs.Image(in_img.name,raw_rec,org_dim)

def default_parametrs():
    return {"num_input":(None,3600),"num_hidden":600,"batch_size":100}	

def compile_autoencoder(hyper_params):
    l_hid,l_out,in_var=build_autoencoder(hyper_params)
    params = lasagne.layers.get_all_params(l_out, trainable=True)
    target_var = T.ivector('targets')
    reconstruction = lasagne.layers.get_output(l_out)
    reduction=lasagne.layers.get_output(l_hid)
    loss = lasagne.objectives.squared_error(reconstruction, in_var).mean()
    #    l1_penalty = regularize_layer_params(l_hid, l1) * 0.001
    #    loss + l1_penalty  
    updates=lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.1, momentum=0.8) 
    return Autoencoder(hyper_params,l_out,in_var,
                         reduction,reconstruction,loss,updates)    

def build_autoencoder(hyper_params=None):
    num_hidden=hyper_params["num_hidden"]
    input_shape = hyper_params["num_input"]
    l_in =  lasagne.layers.InputLayer(shape=input_shape)
    l_hid = lasagne.layers.DenseLayer(l_in, num_units=num_hidden)
    l_out = lasagne.layers.DenseLayer(l_hid, num_units=input_shape[1])
    in_var=l_in.input_var
    return l_hid,l_out,in_var
   
def read_ae(path):
    with open(path, 'r') as f:
        model = pickle.load(f)
    conv_net=compile_autoencoder(model.hyperparams)
    conv_net.set_model(model)
    return conv_net

if __name__ == "__main__": 
    path_dir="../dataset0a/cats"
    ae_path="../dataset0a/ae"
    imgset=np.array(imgs.make_imgs(path_dir,norm=True))
    model=compile_autoencoder(default_parametrs())
    model=train.test_unsuper_model(imgset,model,num_iter=250)
    model.get_model().save(ae_path)
