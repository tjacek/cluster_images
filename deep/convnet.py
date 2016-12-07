import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np
import theano
import theano.tensor as T
import lasagne
import tools 
import pickle
from lasagne.regularization import regularize_layer_params, l2, l1
import deep,train
import utils
import utils.imgs as imgs
import utils.text as text
import utils.data as data
import deep.reader

class Convet(deep.NeuralNetwork):
    def __init__(self,hyperparams,out_layer,preproc,
                     in_var,target_var,
                     features_pred,pred,loss,updates):
        super(Convet,self).__init__(hyperparams,out_layer)
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
    
    def get_category(self,img):
        dist=self.pred(img)
        return [tools.dist_to_category(dist_i) 
                    for dist_i in dist]

def compile_convnet(params,preproc,l1_reg=True):
    in_layer,out_layer,hid_layer,all_layers=build_model(params)
    target_var = T.ivector('targets')
    features_pred = lasagne.layers.get_output(hid_layer)
    pred,in_var=get_prediction(in_layer,out_layer)
    loss=get_loss(pred,in_var,target_var,all_layers,l1_reg)
    updates=get_updates(loss,out_layer)
    return Convet(params,out_layer,preproc, #tools.preprocPost,
                  in_var,target_var,
                  features_pred,pred,loss,updates)

def build_model(params):
    print(params)
    input_shape=params["input_shape"]
    n_filters=params["num_filters"]
    filter_size2D=params["filter_size"]
    pool_size2D=params["pool_size"]
    p_drop=params["p"]
    n_cats=params['n_cats']
    n_hidden=params.get('n_hidden',300) #['n_hidden']

    in_layer = lasagne.layers.InputLayer(
               shape=input_shape)
               #input_var=input_var)
    conv_layer1 = lasagne.layers.Conv2DLayer(
            in_layer, num_filters=n_filters, filter_size=filter_size2D,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    pool_layer1 = lasagne.layers.MaxPool2DLayer(conv_layer1, pool_size=pool_size2D)
    conv_layer2 = lasagne.layers.Conv2DLayer(
            pool_layer1, num_filters=n_filters, filter_size=filter_size2D,
            nonlinearity=lasagne.nonlinearities.rectify)
    pool_layer2 = lasagne.layers.MaxPool2DLayer(conv_layer2, pool_size=pool_size2D)
    dropout = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(pool_layer2, p=p_drop),
            num_units= n_hidden,
            nonlinearity=lasagne.nonlinearities.rectify)
    out_layer = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(dropout, p=p_drop),
            num_units=n_cats,
            nonlinearity=lasagne.nonlinearities.softmax)
    all_layers={"in":in_layer, "conv1":conv_layer1,"pool":pool_layer1,
                "conv2":conv_layer2,"pool2":pool_layer2,
                "hidden":dropout,"out":out_layer }
    return in_layer,out_layer,dropout,all_layers

def get_prediction(in_layer,out_layer):
    in_var=in_layer.input_var
    prediction = lasagne.layers.get_output(out_layer)
    return prediction,in_var

def get_loss(prediction,in_var,target_var,all_layers,l1_reg=True):    
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    l_hid=all_layers["out"]
    if(l1_reg):
        l1_penalty = regularize_layer_params(l_hid, l1) * 0.001
        return loss + l1_penalty
    else:
        return loss

def get_updates(loss,out_layer):
    params = lasagne.layers.get_all_params(out_layer, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.001, momentum=0.9)
    return updates

def default_params():
    return {"input_shape":(None,2,60,60),"num_filters":16,"n_hidden":100,
              "filter_size":(5,5),"pool_size":(4,4),"p":0.5}

def get_model(preproc,nn_path=None,compile=True):
    if(nn_path==None):
        compile=True
    if(compile):
        params=default_params()
        params['n_cats']= data.get_n_cats(y)
        return compile_convnet(params,preproc,False)
    else:
        nn_reader=deep.reader.NNReader(preproc)
        return nn_reader(nn_path,0.1)

if __name__ == "__main__":
    img_path='../dataset2/exp1/train'
    nn_path='../dataset2/exp1/nn_full_nol2'
    preproc=tools.ImgPreproc2D()
    imgset=imgs.make_imgs(img_path,norm=True)
    print("read")
    print(len(imgset))
    x,y=imgs.to_dataset(imgset,data.ExtractCat(),preproc)
    print(x.shape)
    print(y.shape)
    model=get_model(preproc,nn_path,compile=False)
    train.test_super_model(x,y,model,num_iter=120)
    model.get_model().save(nn_path)