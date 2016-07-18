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

class Convet(deep.NeuralNetwork):
    def __init__(self,hyperparams,out_layer,
                     in_var,target_var,
                     features_pred,pred,loss,updates):
        super(Convet,self).__init__(hyperparams,out_layer)
        self.in_var=in_var
        self.target_var=target_var
        self.__features__=theano.function([in_var],features_pred)
        self.pred=theano.function([in_var], pred,allow_input_downcast=True)        
        self.loss=theano.function([in_var,target_var], loss,allow_input_downcast=True)
        self.updates=theano.function([in_var, target_var], loss, 
                               updates=updates,allow_input_downcast=True)

    def features(self,in_img):
        img4D=self.__preproc__(in_img)
        return self.__features__(img4D)
    
    def get_category(self,img):
        dist=self.pred(img)
        return [tools.dist_to_category(dist_i) 
                    for dist_i in dist]

    def get_dim(self):
        dim=self.hyperparams["input_shape"]
        dim=(1,dim[1],dim[2],dim[3])
        return dim

    def __preproc__(self,in_img):
        org_img=in_img.get_orginal()
        img3D=np.expand_dims(org_img,0)
        img4D=np.expand_dims(img3D,0)
        return img4D

def compile_convnet(params,n_cats):
    in_layer,out_layer,hid_layer,all_layers=build_model(params,n_cats)
    target_var = T.ivector('targets')
    features_pred = lasagne.layers.get_output(hid_layer)
    pred,in_var=get_prediction(in_layer,out_layer)
    loss=get_loss(pred,in_var,target_var,all_layers)
    updates=get_updates(loss,out_layer)
    return Convet(params,out_layer,in_var,target_var,
                  features_pred,pred,loss,updates)

def build_model(params,n_cats):
    print(params)
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
    #updates = lasagne.updates.nesterov_momentum(
    #        loss, params, learning_rate=0.001, momentum=0.9)
    updates =lasagne.updates.adagrad(loss,params, learning_rate=0.001)
    return updates

def read_covnet(path):
    with open(path, 'r') as f:
        model = pickle.load(f)
    model.hyperparams["p"]=0.0    
    conv_net=compile_convnet(model.hyperparams,n_cats=10)
    conv_net.set_model(model)
    return conv_net

def default_params():
    return {"input_shape":(None,2,60,60),"num_filters":16,
              "filter_size":(5,5),"pool_size":(4,4),"p":0.5}

if __name__ == "__main__": 
    img_path="../dataset1/cats"
    nn_path="../dataset1/conv_nn"
    imgset=imgs.make_imgs(img_path,norm=True)
    x,y=imgs.to_dataset(imgset,data.ExtractCat(),imgs.to_3D)
    print(x.shape)
    print(y.shape)
    #model=compile_convnet(default_params(),n_cats=10)
    model= read_covnet(nn_path)
    train.test_super_model(x,y,model,num_iter=100)
    model.get_model().save(nn_path)