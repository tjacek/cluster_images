import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import deep
import deep.ae as ae
import numpy as np

def get_autoencoder_extractor(in_path):
    model=deep.read_model(in_path)
    ae_model=ae.build_autoencoder(model.hyperparams)
    ae_model.set_model(model)
    def extractor(data):
    	data=[np.expand_dims(data_i,0) 
    	          for data_i in data]
        return [ae_model.prediction(data_i)
                  for data_i in data]
    return extractor	