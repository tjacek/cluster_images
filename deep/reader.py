import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import deep
import pickle
import deep.convnet 
import deep.ae 
import deep.autoconv
import deep.lstm
import deep.tools

class NNReader(object):
    def __init__(self,preproc=None):
        if(type(preproc)==str):
            self.preproc=get_preproc(preproc)
        else:    
            self.preproc=preproc
        self.types = {'Convet':deep.convnet.compile_convnet,
                      'Autoencoder':deep.ae.compile_autoencoder,
                      'ConvAutoencoder': deep.autoconv.compile_conv_ae,
                      'LSTM':deep.lstm.compile_lstm}

    def __call__(self,in_path, drop_p=0.0,get_hyper=False):
        model=self.__unpickle__(in_path) 
        model.hyperparams['p']=drop_p
        type_reader=self.types[model.type_name]
        neural_net=type_reader(model.hyperparams,self.preproc)
        neural_net.set_model(model)
        if(get_hyper):
            return neural_net,model.hyperparams
        else:
            return neural_net
    
    def __unpickle__(self,in_path):
        with open(str(in_path), 'r') as f:
            model = pickle.load(f)
        return model

def get_preproc(preproc_type):
    if(preproc_type=='time'):
        return deep.tools.ImgPreproc2D()
    return deep.tools.ImgPreprocProj()