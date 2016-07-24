import deep
import utils.imgs as imgs 
import utils.dirs,utils.files
import utils.data as data
import deep.reader as nn_reader
#import deep.autoencoder as ae
import utils.conf
import numpy as np
from seq.features import extract_features

def preproc_seq(conf_dict):
    in_path=conf_dict['in_path']
    nn_path=conf_dict['nn_path']
    extract_features(in_path,nn_path)

if __name__ == "__main__":
    conf_path="conf/dataset1.cfg"
    conf_dict=utils.conf.read_config(conf_path)
    preproc_seq(conf_dict)
