import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import deep
import pickle
import deep.convnet 
import deep.ae 
import deep.autoconv
import deep.lstm
#from deep.convnet import compile_convnet
#from deep.ae import compile_autoencoder
#from deep.autoconv import compile_conv_ae
#from deep.lstm import compile_lstm

class NNReader(object):
    def __init__(self,preproc=None):
        self.preproc=preproc
        self.types = {'Convet':deep.convnet.compile_convnet,
                      'Autoencoder':deep.ae.compile_autoencoder,
                      'ConvAutoencoder': deep.autoconv.compile_conv_ae,
                      'LSTM':deep.lstm.compile_lstm}

    def __call__(self,in_path, drop_p=0.0):
        model=self.__unpickle__(in_path) 
        model.hyperparams['p']=drop_p

        print(model.type_name)
        type_reader=self.types[model.type_name]
        neural_net=type_reader(model.hyperparams,self.preproc)
        neural_net.set_model(model)
        return neural_net
    
    def __unpickle__(self,in_path):
        with open(str(in_path), 'r') as f:
            model = pickle.load(f)
        return model	