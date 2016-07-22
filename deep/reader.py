import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import deep
import pickle
from deep.convnet import compile_convnet

class NNReader(object):
    def __init__(self):
        self.types = {'Convet':compile_convnet}

    def read(self,in_path,determistic=True):
        model=self.__unpickle__(in_path)
        if(determistic):
            model.set_determistic()
        print(model.type_name)
        type_reader=self.types[model.type_name]
        neural_net=type_reader(model.hyperparams)
        neural_net.set_model(model)
        return neural_net
    
    def __unpickle__(self,in_path):
        with open(in_path, 'r') as f:
            model = pickle.load(f)
        return model	