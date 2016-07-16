import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import deep
import deep.ae as ae
import deep.autoconv as ae_conv

import numpy as np

def get_deep_extractor(in_path):
    model=ae_conv.read_conv_ae(in_path)
    #ae_model=ae.build_autoencoder(model.hyperparams)
    #ae_model.set_model(model)
    def extractor(data):
        return [model.prediction(data_i).flatten()
                  for data_i in data]
    return extractor	