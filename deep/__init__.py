import lasagne
import lasagne
import numpy as np
import theano
import theano.tensor as T
import utils.files as files
import deep.tools as tools
import autoencoder as ae

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

def train_model_super(X,y,model,batch_size=100, num_iter=250):
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

def test_model(X,model,batch_size=100, num_iter=250):
    print(X.shape)
    X=to_conv(X,dim=60)
    print(X.shape)
    x_batch,n_batches=tools.get_batch(X,batch_size)
    for epoch in range(num_iter):
        for i in range(n_batches):
            x_i=x_batch[i]
            print(x_i.dtype)
            print(model.loss(x_i))

def to_conv(X,dim=60):
    n_img=X.shape[0]
    X_conv=[np.reshape(X[i],(1,dim,dim)) for i in range(n_img)]
    X_conv=np.array(X_conv)
    return X_conv