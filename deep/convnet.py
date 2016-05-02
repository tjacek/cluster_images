import numpy as np
import theano
import theano.tensor as T
import lasagne
import tools

class Convet(object):
    def __init__(self,l_in,l_out,in_var,target_var,
                     features_pred,pred,loss,updates):
        self.l_in=l_in
        self.l_out=l_out
        self.in_var=in_var
        self.target_var=target_var
        self.features=theano.function([in_var],features_pred)
        self.pred=theano.function([in_var], pred,allow_input_downcast=True)        
        self.loss=theano.function([in_var,target_var], loss,allow_input_downcast=True)
        self.updates=theano.function([in_var, target_var], loss, 
                               updates=updates,allow_input_downcast=True)

    def get_category(self,img):
        dist=self.pred(img)
        return [tools.dist_to_category(dist_i) 
                    for dist_i in dist]

    def get_updates(self):
        return self.updates

def build_convnet(params,n_cats):
    in_layer,out_layer,hid_layer,all_layers=build_model(params,n_cats)
    target_var = T.ivector('targets')
    features_pred = lasagne.layers.get_output(hid_layer)
    pred,in_var=get_prediction(in_layer,out_layer)
    loss=get_loss(pred,in_var,target_var)
    updates=get_updates(loss,out_layer)
    return Convet(in_layer,out_layer,in_var,target_var,
                  features_pred,pred,loss,updates)

def build_model(params,n_cats):
    input_shape=(None, 1, params["dimX"], params["dimY"])
    n_filters=params["num_filters"]
    filter_size2D=(params["filter_size"],params["filter_size"])
    pool_size2D=(params["pool_size"],params["pool_size"])
    p_drop=params["p"]
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
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)
    out_layer = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(dropout, p=p_drop),
            num_units=n_cats,
            nonlinearity=lasagne.nonlinearities.softmax)
    all_layers=[in_layer,conv_layer1,pool_layer1,
                conv_layer2,pool_layer2,dropout,out_layer ]
    return in_layer,out_layer,dropout,all_layers

def get_prediction(in_layer,out_layer):
    in_var=in_layer.input_var
    prediction = lasagne.layers.get_output(out_layer)
    return prediction,in_var

def get_loss(prediction,in_var,target_var):    
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    return loss

def get_updates(loss,out_layer):
    params = lasagne.layers.get_all_params(out_layer, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.001, momentum=0.9)
    return updates

def default_params():
    return {"dimX":60,"dimY":60,"num_filters":16,
              "filter_size":5,"pool_size":2,"p":0.5}
