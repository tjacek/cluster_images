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
import tools 

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
        img4D=self.preproc(in_img)
        raw_rec=self.__reconstructed__(img4D)
        img2D=tools.postproc3D(raw_rec)
        print(img2D.shape)
        return imgs.Image(in_img.name,img2D,in_img.org_dim)

    def __call__(self,in_img):
        img4D=self.preproc(in_img)
        return self.__prediction__(img4D).flatten()

def default_parametrs():
    return {"num_input":(None,2,60,60),"num_hidden":600,"batch_size":100,
            "num_filters":16,"filter_size":(5,5),"pool_size":(4,4)}  

def compile_conv_ae(hyper_params):
    l_hid,l_out,in_var=build_conv_ae(hyper_params)
    params = lasagne.layers.get_all_params(l_out, trainable=True)
    target_var = T.ivector('targets')
    reconstruction = lasagne.layers.get_output(l_out)
    reduction=lasagne.layers.get_output(l_hid)
    loss = lasagne.objectives.squared_error(reconstruction, in_var).mean()
    l1_penalty = regularize_layer_params(l_hid, l1) * 0.0001
    loss+=l1_penalty  
    updates=lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.001, momentum=0.8) 
    return ConvAutoencoder(hyper_params,l_out,tools.preproc3D,in_var,
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
    print(pool_size)
    flat_layer1=lasagne.layers.ReshapeLayer(pool_layer1,([0],-1) )
    flat_size=lasagne.layers.get_output_shape(flat_layer1)[1]
    print 'flat size %d' % flat_size
    hidden=lasagne.layers.DenseLayer(flat_layer1,
             num_units=num_hidden,
             nonlinearity=lasagne.nonlinearities.rectify)

    flat_layer2=lasagne.layers.DenseLayer(hidden,
             num_units=flat_size,
             nonlinearity=lasagne.nonlinearities.rectify)

    pool_layer2=lasagne.layers.ReshapeLayer(flat_layer2,(pool_size))

    conv_layer2=lasagne.layers.Upscale2DLayer(pool_layer2, pool_size2D)
    
    out_layer= lasagne.layers.TransposedConv2DLayer(
            conv_layer2, num_filters=2, filter_size=filter_size2D,
            nonlinearity=None,#lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    
    in_var=l_in.input_var
    print(lasagne.layers.get_output_shape( out_layer))
    return hidden,out_layer,in_var#l_hid,l_out,in_var
   
def read_conv_ae(path):
    with open(path, 'r') as f:
        model = pickle.load(f)
    #nn.layers.set_all_param_values(model, data)
    conv_net=compile_autoencoder(model.hyperparams)
    conv_net.set_model(model)
    return conv_net

if __name__ == "__main__": 
    path_dir="../dataset1/cats"
    ae_path="../dataset1/conv_ae_"
    imgset=imgs.make_imgs(path_dir,norm=True,transform=imgs.to_3D)
    print(imgset.shape)
    #model= read_conv_ae(ae_path)
    nn_reader=deep.reader.NNReader()
    model= nn_reader.read(ae_path)
    #model=compile_conv_ae(default_parametrs())
    model=deep.train.test_unsuper_model(imgset,model,num_iter=500)
    model.get_model().save(ae_path)