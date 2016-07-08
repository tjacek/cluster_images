import lasagne,pickle
import numpy as np
import theano
import theano.tensor as T
import utils.files as files
import deep.tools as tools
import autoencoder as ae

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

def train_model_unsuper(imgs,hyper_params,num_iter=500,input_dim=(60,60)):
    batch_size=hyper_params["batch_size"]
    input_dim=(np.product(imgs[0].shape),) 
    hyper_params["num_input"]=np.product(input_dim)
    model=ae.Autoencoder(hyper_params)
    input_var=model.get_input_var()
    updates=model.get_updates()
    train_fn = theano.function([input_var], model.loss, updates=updates)
    #(hyper_params["num_input"],)
    print(input_dim)
    print("Input img size:" + str(imgs[0].shape))
    #imgs=[img_i.reshape(input_dim) for img_i in imgs]
    imgs=[img_i.reshape(input_dim) for img_i in imgs]
    batch,n_batches=tools.get_batch(imgs,batch_size)
    print("Number of batches " + str(len(batch)))
    print("Number of iters " + str(num_iter))
    for epoch in range(num_iter):
        cost_e = []
        for i in range(n_batches):
            img_i=batch[i]
            loss_i=train_fn(img_i)
            cost_e.append(loss_i)
        cost_mean=np.mean(cost_e)
        print(str(epoch) + " "+str(cost_mean))
    return model

def train_model_super(X,y,model,batch_size=100, num_iter=5):
    X=X.reshape((X.shape[0],X.shape[1]*X.shape[2]))
    input_var=model.get_input_var()
    updates=model.get_updates()
    train_fn = theano.function([input_var,model.target_var], model.loss, updates=updates,allow_input_downcast=True)
    x_batch,n_batches=tools.get_batch(X,batch_size)
    y_batch,n_batches=tools.get_batch(y,batch_size)
    print("Number of batches " + str(len(x_batch)))
    print("Number of iters " + str(num_iter))
    for epoch in range(num_iter):
        cost_e = []
        for i in range(n_batches):
            x_i=x_batch[i]
            y_i=y_batch[i]
            loss_i= train_fn(x_i,y_i)
            cost_e.append(loss_i)
        cost_mean=np.mean(cost_e)
        print(str(epoch) + " "+str(cost_mean))
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