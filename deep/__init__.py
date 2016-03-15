import lasagne
import lasagne
import numpy as np
import theano
import theano.tensor as T
import utils.files as files
import deep.tools as tools
import autoencoder as ae

def train_model_unsuper(imgs,hyper_params,num_iter=1000):
    batch_size=hyper_params["batch_size"]
    model=ae.Autoencoder(hyper_params)
    input_var=model.get_input_var()
    updates=model.get_updates()
    train_fn = theano.function([input_var], model.loss, updates=updates)
    input_dim=(hyper_params["num_input"],)
    imgs=[img_i.reshape(input_dim) for img_i in imgs]
    batch,n_batches=tools.get_batch(imgs,batch_size)
    print("Number of batches " + str(len(batch)))
    for epoch in range(num_iter):
        cost_e = []
        for i in range(n_batches):
            img_i=batch[i]
            loss_i=train_fn(img_i)
            cost_e.append(loss_i)
        cost_mean=np.mean(cost_e)
        print(str(epoch) + " "+str(cost_mean))
    return model

def train_model_super(X,y,model,batch_size=100, num_iter=1000):
    X=X.reshape((X.shape[0],X.shape[1]*X.shape[2]))
    input_var=model.get_input_var()
    updates=model.get_updates()
    train_fn = theano.function([input_var,model.target_var], model.loss, updates=updates,allow_input_downcast=True)
    #pred_fn = theano.function([input_var], model.prediction)
    x_batch,n_batches=tools.get_batch(X,batch_size)
    y_batch,n_batches=tools.get_batch(y,batch_size)
    print("Number of batches " + str(len(x_batch)))
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