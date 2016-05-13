import lasagne
import numpy as np
import theano
import theano.tensor as T
import utils.files as files
import deep.tools as tools
import autoencoder
import pickle
import convnet

class StackedAE(object):
    def __init__(self,params,l_in,l_out,in_var,target_var,
                 features_pred,pred,loss,updates):
        self.hyperparams=params
        self.l_in=l_in
        self.l_out=l_out
        self.in_var=in_var
        self.target_var=target_var
        self.features=theano.function([in_var],features_pred,allow_input_downcast=True)
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
        return convnet.Model(self.hyperparams,data)
    
    def set_model(self,model):
        lasagne.layers.set_all_param_values(self.l_out,model.params)

    def get_dim(self):
        dim=self.hyperparams["input_shape"]
        dim=(1,dim[1],dim[2],dim[3])
        return dim

    def __str__(self):
        return str(self.hyperparams)

def build_sae(params,n_cats):
    l_in,l_hid,l_out=build_model(params,n_cats)
    target_var = T.ivector('targets')
    prediction,in_var=get_prediction(l_in,l_out)
    features_pred=get_features(l_hid)
    loss=get_loss(prediction,target_var)
    updates=get_updates(loss,l_out)
    return StackedAE(params,l_in,l_out,in_var,target_var,
                features_pred,prediction,loss,updates)

def build_model(params,n_cats):
    ae_path=params["auto_path"]
    ae_model=autoencoder.read_ae(ae_path)
    W_hid,b_hid=ae_model.get_numpy()
    input_size,hidden_size=W_hid.shape
    print(type(W_hid))
    #print(hidden_size)
    l_in =  lasagne.layers.InputLayer(shape=(None,input_size),W=W_hid,b=b_hid)#,input_var=self.input_var)
    l_hid = lasagne.layers.DenseLayer(l_in, num_units=hidden_size)
    softmax = lasagne.nonlinearities.softmax
    l_out = lasagne.layers.DenseLayer(l_hid,n_cats, nonlinearity=softmax)
    return l_in,l_hid,l_out

def get_prediction(in_layer,out_layer):
    in_var=in_layer.input_var
    prediction = lasagne.layers.get_output(out_layer)
    return prediction,in_var

def get_loss(prediction,target_var):    
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    return loss

def get_updates(loss,out_layer):
    params = lasagne.layers.get_all_params(out_layer, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.001, momentum=0.9)
    return updates

def get_features(l_hid):
    features_pred = lasagne.layers.get_output(l_hid)
    return features_pred

def read_sae(path):
    with open(path, 'r') as f:
        model = pickle.load(f)
    #model.hyperparams["p"]=0.0    
    #nn.layers.set_all_param_values(model, data)
    sae=build_convnet(model.hyperparams,n_cats=10)
    sae_net.set_model(model)
    return sae_net