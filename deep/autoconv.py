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
from lasagne.layers.conv import TransposedConv2DLayer

class ConvAutoencoder(deep.NeuralNetwork):
    def __init__(self,hyperparams,out_layer,in_var,
                     reduction,reconstruction,loss,updates):
        super(ConvAutoencoder,self).__init__(hyperparams,out_layer)
        self.prediction=theano.function([in_var], reduction,allow_input_downcast=True)
        self.reconstructed=theano.function([in_var], 
                                           reconstruction,allow_input_downcast=True)
        self.loss=theano.function([in_var], loss,allow_input_downcast=True,name="loss")
        self.updates=theano.function([in_var], loss, 
                               updates=updates,allow_input_downcast=True)

def default_parametrs():
    return {"num_input":(None,1,60,60),"num_hidden":600,"batch_size":100,
            "num_filters":4,"filter_size":(5,5),"pool_size":(8,8)}  

def compile_autoencoder(hyper_params):
    l_hid,l_out,in_var=build_conv_ae(hyper_params)
    params = lasagne.layers.get_all_params(l_out, trainable=True)
    target_var = T.ivector('targets')
    reconstruction = lasagne.layers.get_output(l_out)
    reduction=lasagne.layers.get_output(l_hid)
    loss = lasagne.objectives.squared_error(reconstruction, in_var).mean()
    #    l1_penalty = regularize_layer_params(l_hid, l1) * 0.001
    #    loss + l1_penalty  
    updates=lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.001, momentum=0.8) 
    return ConvAutoencoder(hyper_params,l_out,in_var,
                         reduction,reconstruction,loss,updates)    

def build_conv_ae(hyper_params):
    num_hidden=hyper_params["num_hidden"]
    input_shape = hyper_params["num_input"]
    n_filters = hyper_params["num_filters"]
    filter_size2D = hyper_params["filter_size"]
    pool_size2D =  hyper_params["pool_size"]
    l_in = lasagne.layers.InputLayer(
               shape=input_shape)
    conv_layer1 = lasagne.layers.Conv2DLayer(
            l_in, num_filters=n_filters, filter_size=filter_size2D,
            nonlinearity=None,#lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    pool_layer1 = lasagne.layers.MaxPool2DLayer(conv_layer1, pool_size=pool_size2D)
    pool_size=list(lasagne.layers.get_output_shape(pool_layer1))
    pool_size[0]=[0]
    pool_size=tuple(pool_size)

    flat_layer1=lasagne.layers.ReshapeLayer(pool_layer1,([0],-1) )
    flat_size=lasagne.layers.get_output_shape(flat_layer1)[1]

    hidden=lasagne.layers.DenseLayer(flat_layer1,
             num_units=num_hidden,
             nonlinearity=lasagne.nonlinearities.rectify)

    flat_layer2=lasagne.layers.DenseLayer(hidden,
             num_units=flat_size,
             nonlinearity=lasagne.nonlinearities.rectify)

    pool_layer2=lasagne.layers.ReshapeLayer(flat_layer2,(pool_size))

    conv_layer2=lasagne.layers.Upscale2DLayer(pool_layer2, pool_size2D)
    
    out_layer= lasagne.layers.TransposedConv2DLayer(
            conv_layer2, num_filters=1, filter_size=filter_size2D,
            nonlinearity=None,#lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    #features=lasagne.layers.get_output(hidden)
    #reconstr=lasagne.layers.get_output(out_layer)
    in_var=l_in.input_var
    return hidden,out_layer,in_var#l_hid,l_out,in_var
   
def read_ae(path):
    with open(path, 'r') as f:
        model = pickle.load(f)
    #nn.layers.set_all_param_values(model, data)
    conv_net=compile_autoencoder(model.hyperparams)
    conv_net.set_model(model)
    return conv_net

if __name__ == "__main__": 
    path_dir="../dataset0a/cats"
    imgset=imgs.make_imgs(path_dir,norm=False,conv=True)
    print(imgset.shape)
    model=compile_autoencoder(default_parametrs())
    model=deep.test_unsuper_model(imgset,model,num_iter=10)
    model.get_model().save("../dataset0a/ae")