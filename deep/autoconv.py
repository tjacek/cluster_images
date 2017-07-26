import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import lasagne,pickle
import numpy as np
import theano
import theano.tensor as T
import utils.paths.files as files
import utils.imgs as imgs
import deep,convnet
from lasagne.regularization import regularize_layer_params, l2, l1
from lasagne.layers.conv import TransposedConv2DLayer
import tools 
import utils.data as data
import utils.actions 
import train

class ConvAutoencoder(deep.NeuralNetwork):
    def __init__(self,hyperparams,out_layer,preproc,in_var,
                     reduction,reconstruction,loss,updates):
        super(ConvAutoencoder,self).__init__(hyperparams,out_layer)
        self.preproc=preproc
        self.__prediction__=theano.function([in_var], reduction,allow_input_downcast=True)
        self.__reconstructed__=theano.function([in_var], 
                                           reconstruction,allow_input_downcast=True)
        self.loss=theano.function([in_var], loss,allow_input_downcast=True,name="loss")
        self.updates=theano.function([in_var], loss, 
                               updates=updates,allow_input_downcast=True)

    def reconstructed(self,in_img):
        img4D=self.preproc.apply(in_img)
        raw_rec=self.__reconstructed__(img4D)
        img2D=tools.postproc3D(raw_rec)
        return imgs.Image(in_img.name,img2D,in_img.org_dim)

    def __call__(self,in_img):
        img4D=self.preproc.apply(in_img)
        return self.__prediction__(img4D).flatten()

def compile_conv_ae(hyper_params,preproc):
    l_hid,l_out,in_var=build_conv_ae(hyper_params)
    params = lasagne.layers.get_all_params(l_out, trainable=True)
    target_var = T.ivector('targets')
    reconstruction = lasagne.layers.get_output(l_out)
    reduction=lasagne.layers.get_output(l_hid)
    
    loss = lasagne.objectives.squared_error(reconstruction, in_var).mean()
    l1_penalty = regularize_layer_params(l_hid, l1) * 0.0001
    #loss+=l1_penalty  
    updates=lasagne.updates.adadelta(loss, params, learning_rate=0.001) 
    #lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.001, momentum=0.8) 
    return ConvAutoencoder(hyper_params,l_out,preproc,in_var,
                         reduction,reconstruction,loss,updates)    

def build_conv_ae(hyper_params):
    l_in = lasagne.layers.InputLayer(hyper_params['input_shape'])
    l_conv1 = lasagne.layers.Conv2DLayer(l_in,
                num_filters=hyper_params['n_filters1'],
                filter_size=hyper_params['filter_size1'],
                pad='same',
                nonlinearity=lasagne.nonlinearities.rectify,
                #W=lasagne.init.GlorotUniform(),
                name='conv1')
    show_layer(l_conv1)
    l_pool1 = lasagne.layers.MaxPool2DLayer(l_conv1, pool_size=hyper_params['pool_size1'],name='pool1')
    show_layer(l_pool1)
    l_conv2 = lasagne.layers.Conv2DLayer(l_pool1,
                num_filters=hyper_params['n_filters2'],
                filter_size=hyper_params['filter_size1'],
                pad='same',
                nonlinearity=lasagne.nonlinearities.rectify,
                #W=lasagne.init.GlorotUniform(),
                name='conv2')
    show_layer(l_conv2)
    
    l_pool2 = lasagne.layers.MaxPool2DLayer(l_conv2, pool_size=hyper_params['pool_size1'],name='pool2')
    show_layer(l_pool2)

    l_conv3 = lasagne.layers.Conv2DLayer(l_pool2,
                num_filters=hyper_params['n_filters3'],
                filter_size=hyper_params['filter_size1'],
                pad='same',
                nonlinearity=lasagne.nonlinearities.rectify,
                #W=lasagne.init.GlorotUniform(),
                name='conv3')
    show_layer(l_conv3)

    l_pool3 = lasagne.layers.MaxPool2DLayer(l_conv3, pool_size=hyper_params['pool_size1'])
    show_layer(l_pool3)

    l_hidden = lasagne.layers.DenseLayer(l_pool3, num_units=hyper_params['num_hidden'],
                      nonlinearity=lasagne.nonlinearities.rectify,name='hidden')
    show_layer(l_hidden)
    
    l_hidden_inv = lasagne.layers.InverseLayer(l_hidden, l_hidden,name='hidden_inv')
    show_layer(l_hidden_inv)

    l_pool3_inv = lasagne.layers.Upscale2DLayer(l_hidden_inv, 
                        scale_factor=hyper_params['pool_size1'],
                        name='pool3_inv')
    show_layer(l_pool3_inv)
    
    l_conv3_inv = lasagne.layers.InverseLayer(l_pool3_inv, l_conv3,name='l_conv3_inv')
    show_layer(l_conv3_inv)

    l_pool2_inv = lasagne.layers.Upscale2DLayer(l_conv3_inv, 
                          scale_factor=hyper_params['pool_size1'],name='l_pool2_inv')
    show_layer(l_pool2_inv)

    l_conv2_inv = lasagne.layers.InverseLayer(l_pool2_inv, l_conv2,name='l_conv2_inv')
    show_layer(l_conv2_inv)
    
    l_pool1_inv = lasagne.layers.Upscale2DLayer(l_conv2_inv, 
                               scale_factor=hyper_params['pool_size1'],name='l_pool1_inv')
    show_layer(l_pool1_inv)

    l_conv1_inv = lasagne.layers.InverseLayer(l_pool1_inv, l_conv1,name='l_conv1_inv')
    show_layer(l_conv1_inv)

    l_out = l_conv1_inv

    in_var=l_in.input_var
    print("Out layer shape")
    print(lasagne.layers.get_output_shape( l_out))
    return l_hidden,l_out,in_var

def conv_ae_params(num_hidden=100,n_frames=2):
    return {'input_shape':(None, n_frames, 64, 64),
            'n_filters1':16,
            'n_filters2':8,
            'n_filters3':8, 
            'filter_size1':(3, 3),
            'pool_size1':(2,2),
            'num_hidden':num_hidden}

def show_layer(layer):
    print(layer.name)
    print(lasagne.layers.get_output_shape(layer))

def train_ae(img_path,nn_path):
    preproc=tools.ImgPreproc2D()
    imgset=imgs.make_imgs(img_path,norm=True,transform=preproc)

    x=imgs.to_dataset(imgset,extract_cat=None,transform=None)

    model=get_model(preproc,compile=False)
    train.test_unsuper_model(x,model,num_iter=50)
    model.get_model().save(nn_path)

def reconstruct(img_path,nn_path,out_path):
    preproc=deep.tools.ImgPreproc2D()
    nn_reader=deep.reader.NNReader(preproc)
    conv_ae=nn_reader(nn_path) 
    def ae_transform(in_img):
        return conv_ae.reconstructed(in_img)
    utils.actions.transform_actions(img_path,out_path,ae_transform,seq_transform=False)

def get_model(preproc,compile=True):
    if(compile):
        params=conv_ae_params(num_hidden=100,n_frames=2)
        return compile_conv_ae(params,preproc)
    else:  
        nn_reader=deep.reader.NNReader(preproc)
        return nn_reader(nn_path,0.0)

if __name__ == "__main__": 
    img_path="../../AArtyk/MSR/proj"
    nn_path="../../AArtyk/conv_ae"
    out_path="../../AArtyk/proj_rec"
    reconstruct(img_path,nn_path,out_path)