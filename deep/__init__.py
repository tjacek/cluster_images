import lasagne,pickle
import numpy as np
import theano
import theano.tensor as T
import utils.files as files
import deep.tools as tools
#import ae

class NeuralNetwork(object):
    def __init__(self,hyperparams,params):
        self.hyperparams=hyperparams
        self.params=params

    def get_model(self):
        data = lasagne.layers.get_all_param_values(self.l_out)
        return deep.Model(self.hyperparams,data)
    
    def set_model(self,model):
        lasagne.layers.set_all_param_values(self.l_out,model.params)

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

def show_cats(X,y,model,transform,
                      batch_size=100,num_iter=5):
    X=transform(X,dim=60)
    x_batch,n_batches=tools.get_batch(X,1)
    for i,y_i in enumerate(y):
        cat_i=model.get_category(x_batch[i])
        print(cat_i)
        print(y_i)

def test_super_model(X,y,model,transform,
                      batch_size=100,num_iter=500):
    print("Num iters " + str(num_iter))
    X=transform(X,dim=60)
    x_batch,n_batches=tools.get_batch(X,batch_size)
    y_batch,n_batches=tools.get_batch(y,batch_size)
    for epoch in range(num_iter):
        cost_e = []
        for i in range(n_batches):
            x_i=x_batch[i]
            if((i%25)==0):
                print(model.get_category(x_i))
            y_i=y_batch[i]
            loss_i=model.updates(x_i,y_i)
            cost_e.append(loss_i)
        cost_mean=np.mean(cost_e)
        print(str(epoch) + " "+str(cost_mean))
    return model

def test_unsuper_model(X,model,transform=None,
                        batch_size=100,num_iter=250):
    print(X.shape)
    if(transform!=None):
        X=transform(X,dim=60)
    print(X.shape)
    x_batch,n_batches=tools.get_batch(X,batch_size)
    for epoch in range(num_iter):
        cost_e = []
        for i in range(n_batches):
            x_i=x_batch[i]
            loss_i=model.updates(x_i)
            cost_e.append(loss_i)
        cost_mean=np.mean(cost_e)
        print(str(epoch) + " "+str(cost_mean))
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