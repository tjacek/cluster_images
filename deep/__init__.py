import lasagne,pickle
import numpy as np
import theano
import theano.tensor as T
import utils.files as files

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