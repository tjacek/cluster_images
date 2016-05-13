import numpy as np
import theano
import theano.tensor as T
import lasagne
import tools
import pickle
from lasagne.regularization import regularize_layer_params, l2, l1

class Model(object):
    def __init__(self,hyperparams,params):
        self.params=params
        self.hyperparams=hyperparams

class Convet(object):
    def __init__(self,params,l_in,l_out,in_var,target_var,
                     features_pred,pred,loss,updates):
        self.hyperparams=params
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

    def get_model(self):
        data = lasagne.layers.get_all_param_values(self.l_out)
        return Model(self.hyperparams,data)
    
    def set_model(self,model):
        lasagne.layers.set_all_param_values(self.l_out,model.params)

    def get_dim(self):
        dim=self.hyperparams["input_shape"]
        dim=(1,dim[1],dim[2],dim[3])
        return dim

    def __str__(self):
        return str(self.hyperparams)

def build_convnet(params,n_cats):
    in_layer,out_layer,hid_layer,all_layers=build_model(params,n_cats)
    target_var = T.ivector('targets')
    features_pred = lasagne.layers.get_output(hid_layer)
    pred,in_var=get_prediction(in_layer,out_layer)
    loss=get_loss(pred,in_var,target_var,all_layers)
    updates=get_updates(loss,out_layer)
    return Convet(params,in_layer,out_layer,in_var,target_var,
                  features_pred,pred,loss,updates)

def build_model(params,n_cats):
    input_shape=params["input_shape"]
    n_filters=params["num_filters"]
    filter_size2D=params["filter_size"]
    pool_size2D=params["pool_size"]
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
            num_units=300,
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

def get_loss(prediction,in_var,target_var,all_layers):    
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    l_hid=all_layers["out"]
    l1_penalty = regularize_layer_params(l_hid, l1) * 0.001
    return loss + l1_penalty

def get_updates(loss,out_layer):
    params = lasagne.layers.get_all_params(out_layer, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.001, momentum=0.9)
    return updates

def save_covnet(conv_net,path):
    data = conv_net.get_model()#lasagne.layers.get_all_param_values(conv_net.l_out)
    with open(path, 'w') as f:
        pickle.dump(data, f)

def read_covnet(path):
    with open(path, 'r') as f:
        model = pickle.load(f)
    model.hyperparams["p"]=0.0    
    #nn.layers.set_all_param_values(model, data)
    conv_net=build_convnet(model.hyperparams,n_cats=10)
    conv_net.set_model(model)
    return conv_net

def default_params():
    return {"input_shape":(None,2,60,60),"num_filters":16,
              "filter_size":(5,5),"pool_size":(4,4),"p":0.5}
