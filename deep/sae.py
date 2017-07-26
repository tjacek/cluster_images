import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import lasagne
import numpy as np
import theano
import theano.tensor as T
import deep.tools #as tools
import pickle
import convnet
import deep
import deep.reader
import utils.data as data
import utils.imgs as imgs

class ConvSAE(deep.NeuralNetwork):
    def __init__(self,hyperparams,out_layer,preproc,
                     in_var,target_var,
                     features_pred,pred,loss,updates):
        super(ConvSAE,self).__init__(hyperparams,out_layer)
        self.preproc=preproc
        self.in_var=in_var
        self.target_var=target_var
        self.__features__=theano.function([in_var],features_pred)
        self.pred=theano.function([in_var], pred,allow_input_downcast=True)        
        self.loss=theano.function([in_var,target_var], loss,allow_input_downcast=True)
        self.updates=theano.function([in_var, target_var], loss, 
                               updates=updates,allow_input_downcast=True)

    def __call__(self,in_img):
        img4D=self.preproc.apply(in_img)
        return self.__features__(img4D).flatten()
    
    def __getitem__(self,key):
        return self.hyperparams[key]

    def get_category(self,img):
        dist=self.get_distribution(img)
        return np.argmax(dist)#[tools.dist_to_category(dist_i) 
               #     for dist_i in dist]

    def get_distribution(self,x):
        if(len(x.shape)!=4):
            img4D=self.preproc.apply(x)
        else:
            img4D=x
        #x.name=str(x.name)
        img_x=self.pred(img4D).flatten()
        return img_x

    def dim(self):
        return self.hyperparams['n_hidden']

def compile_conv_ae(hyper_params,preproc):
    all_layers=build_conv_sae(hyper_params)
    target_var = T.ivector('targets')
    features_pred = make_features(all_layers)
    pred,in_var=get_prediction(all_layers)
    loss=get_loss(pred,in_var,target_var,all_layers)
    updates=get_updates(loss,all_layers)
    add_sae_adnot(hyper_params,all_layers)
    return ConvSAE(hyper_params, all_layers['out'],
                   preproc,in_var,target_var,
                   features_pred,pred,loss,updates)
    

def build_conv_sae(hyper_params):
    l_in = lasagne.layers.InputLayer(hyper_params['input_shape'])
    l_conv1 = lasagne.layers.Conv2DLayer(l_in,
                num_filters=hyper_params['n_filters1'],
                filter_size=hyper_params['filter_size1'],
                pad='same',
                nonlinearity=lasagne.nonlinearities.rectify,
                name='conv1')

    l_pool1 = lasagne.layers.MaxPool2DLayer(l_conv1, pool_size=hyper_params['pool_size1'],name='pool1')

    l_conv2 = lasagne.layers.Conv2DLayer(l_pool1,
                num_filters=hyper_params['n_filters2'],
                filter_size=hyper_params['filter_size1'],
                pad='same',
                nonlinearity=lasagne.nonlinearities.rectify,
                name='conv2')
    
    l_pool2 = lasagne.layers.MaxPool2DLayer(l_conv2, pool_size=hyper_params['pool_size1'],name='pool2')

    l_conv3 = lasagne.layers.Conv2DLayer(l_pool2,
                num_filters=hyper_params['n_filters3'],
                filter_size=hyper_params['filter_size1'],
                pad='same',
                nonlinearity=lasagne.nonlinearities.rectify,
                name='conv3')

    l_pool3 = lasagne.layers.MaxPool2DLayer(l_conv3, pool_size=hyper_params['pool_size1'])
    
    
    hidden = lasagne.layers.DenseLayer(
            l_pool3,#lasagne.layers.dropout(l_pool3, p=p_drop),
            num_units= hyper_params['n_hidden'],
            nonlinearity=lasagne.nonlinearities.rectify,
            name='hidden')
    out_layer = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(hidden, p=hyper_params['p_drop']),
            num_units= hyper_params['n_cats'],
            nonlinearity=lasagne.nonlinearities.softmax)
    
    all_layers={ "in":l_in, 
                 "conv1":l_conv1,"pool1":l_pool1,
                 "conv2":l_conv2,"pool2":l_pool2,
                 "conv3":l_conv3,"pool3":l_pool3,
                 "hidden":hidden,"out":out_layer }
    return all_layers#in_layer,out_layer,dropout,all_layers

def make_features(all_layers):
    hid_layer=all_layers['hidden']
    return lasagne.layers.get_output(hid_layer)

def get_prediction(all_layers):
    in_var=all_layers['in'].input_var
    prediction = lasagne.layers.get_output(all_layers['out'])
    return prediction,in_var

def get_loss(prediction,in_var,target_var,all_layers):    
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    #l_hid=all_layers["out"]
    return loss

def get_updates(loss,all_layers):
    out_layer=all_layers['out']
    params = lasagne.layers.get_all_params(out_layer, trainable=True)
    updates =lasagne.updates.adadelta(loss, params, learning_rate=0.001) 
    return updates

def add_sae_adnot(hyper_params,all_layers):
    l_out=all_layers['out']
    hyper_params['out_W']=(hyper_params['n_hidden'],hyper_params['n_cats'])#tuple(w_shape)
    hyper_params['out_b']=(hyper_params['n_cats'])  #tuple(b_shape)
    return hyper_params 

def conv_ae_params(n_cats=20,num_hidden=100,n_frames=2):
    return {'input_shape':(None, n_frames, 64, 64),
            'n_filters1':16,
            'n_filters2':8,
            'n_filters3':8, 
            'filter_size1':(3, 3),
            'pool_size1':(2,2),
            'n_hidden':num_hidden,
            'n_cats':n_cats,
            'p_drop':0.2}

def make_model(in_path,hyper_params):
    preproc=deep.tools.ImgPreproc2D()
    conv_sae=compile_conv_ae(hyper_params,preproc)

    with open(str(in_path), 'r') as f:
        model = pickle.load(f)
        model.add_empty_params(conv_sae['out_W'])
        model.add_empty_params(conv_sae['out_b'])
        conv_sae.set_model(model)
        return conv_sae


if __name__ == "__main__":
    in_path="../../AArtyk/ae/conv_ae"
    hyper_params=conv_ae_params()
    conv_sea=make_model(in_path,hyper_params)

    nn_path="../../AArtyk/nn_sae"
    img_path="../../AArtyk/train"

    preproc=deep.tools.ImgPreproc2D()
    imgset=imgs.make_imgs(img_path,norm=True)
    extract_cat=data.ExtractCat()
    x,y=imgs.to_dataset(imgset,extract_cat,preproc)
    train.test_super_model(x,y,conv_sea,num_iter=10)
    model.get_model().save(nn_path)
