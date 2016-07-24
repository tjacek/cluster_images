import deep
import utils.imgs as imgs 
import utils.dirs,utils.files
import utils.data as data
import deep.reader 
import utils.conf
import numpy as np
#from seq.features import extract_features
import basic.reduction
import basic.external
from basic.external import external_features

def transform_features(conf_dict):
    in_path=conf_dict['in_path']
    out_path=conf_dict['out_path']
    extractor=select_extractor(conf_dict)
    basic.external.transform_features(in_path,out_path,extractor) 

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
    elif extractor_type=='text':
        text_path=conf_dict['text_path']
        feat_dict=basic.external.read_external(text_path)
        extractor=lambda img_i:feat_dict[img_i.name]
    else:
        extractor=getattr(basic.reduction,extractor_type)
    return extractor

if __name__ == "__main__":
    conf_path="conf/dataset1_.cfg"
    conf_dict=utils.conf.read_config(conf_path)
    #make_features(conf_dict)
    transform_features(conf_dict)