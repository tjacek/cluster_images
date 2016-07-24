import deep
import utils.imgs as imgs 
import utils.dirs,utils.files
import utils.data as data
import deep.reader 
import utils.conf
import numpy as np
#from seq.features import extract_features
import basic.reduction
from basic.external import external_features

#EXTRACTORS={'spectral':basic.redu,'pca'}

def preproc_seq(conf_dict):
    in_path=conf_dict['in_path']
    extractor=select_extractor(conf_dict)
    extract_features(in_path,extractor)

def make_features(conf_dict):
    in_path=conf_dict['in_path']
    out_path=conf_dict['out_path']
    extractor=select_extractor(conf_dict)
    data=imgs.make_imgs(in_path,norm=True)
    print(type(data[0]))
    external_features(out_path,data,extractor)

def select_extractor(conf_dict):
    extractor_type=conf_dict['extractor']
    if(extractor_type=='deep'):
        nn_path=conf_dict['nn_path']
        nn_reader=deep.reader.NNReader()
        extractor=nn_reader.read(nn_path)
    else:
         extractor=getattr(basic.reduction,extractor_type)
    return extractor

if __name__ == "__main__":
    conf_path="conf/dataset1.cfg"
    conf_dict=utils.conf.read_config(conf_path)
    #print(getattr(basic.reduction,'transform_pca') )
    make_features(conf_dict)
