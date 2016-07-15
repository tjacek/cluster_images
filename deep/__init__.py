import lasagne,pickle
import numpy as np
import theano
import theano.tensor as T
import utils.files as files
#import deep.tools as tools

class NeuralNetwork(object):
    def __init__(self,hyperparams,out_layer):
        self.hyperparams=hyperparams
        self.out_layer=out_layer

    def get_model(self):
        data = lasagne.layers.get_all_param_values(self.out_layer)
        return Model(self.hyperparams,data)
    
    def set_model(self,model):
        lasagne.layers.set_all_param_values(self.out_layer,model.params)

    def __str__(self):
        return str(self.hyperparams)

class Model(object):
    def __init__(self,hyperparams,params):
        self.params=params
        self.hyperparams=hyperparams

    def save(self,path):
        with open(path, 'w') as f:
            pickle.dump(self, f)

def read_model(in_path):
    with open(in_path, 'r') as f:
        model = pickle.load(f)
    return model

def to_1D(X,dim=7200):
    n_img=X.shape[0]
    print(X[0].shape)
    X_conv=[X[i].flatten() for i in range(n_img)]
    X_conv=np.array(X_conv)
    
    return X_conv

def to_conv(X,dim=60):
    n_img=X.shape[0]
    X_conv=[np.reshape(X[i]) for i in range(n_img)]
    X_conv=np.array(X_conv)
    return X_conv

def to_vol(X,dim=60):
    n_img=X.shape[0]
    def reshape_img(img_i):
        #print(img_i.shape)
        x_i=np.reshape(img_i,(2*dim,dim))
        #print(x_i.shape)
        img1,img2=split_img(x_i)
        
        x_i=np.array([img1,img2])
        #print(x_i.shape)
        return x_i
    X_conv=[reshape_img(X[i]) for i in range(n_img)]
    X_conv=np.array(X_conv)
    return X_conv
    
def split_img(img):
    img_height=img.shape[0]/2
    img1=img[...][0:img_height]
    img2=img[...][img_height:2*img_height]  
    return img1,img2