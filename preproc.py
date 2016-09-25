import deep
import utils.imgs as imgs 
import utils.dirs,utils.files
import utils.data as data
import deep.reader 
import deep.tools 
import utils.conf
import numpy as np
import basic
import basic.reduction
import basic.external
import basic.combine
from basic.external import external_features

def transform_features(conf_dict):
    in_path=conf_dict['in_path']
    out_path=conf_dict['out_path']
    extractor=select_extractor(conf_dict)
    basic.external.transform_features(in_path,out_path,extractor) 

def make_features(conf_dict):
    in_path=conf_dict['img_path']
    out_path=conf_dict['feat_path']
    extractor=select_extractor(conf_dict)
    data=imgs.make_imgs(in_path,norm=True)
    print(type(data[0]))
    external_features(out_path,data,extractor)

def select_extractor(conf_dict):
    extractor_type=conf_dict['extractor']
    preproc3D=deep.tools.ImgPreproc()
    if(extractor_type=='deep'):
        nn_path=conf_dict['nn_path']
        nn_reader=deep.reader.NNReader()
        extractor=nn_reader.read(nn_path,preproc3D)
    elif extractor_type=='text':
        text_path=conf_dict['text_path']
        feat_dict=basic.external.read_external(text_path)
        extractor=lambda img_i:feat_dict[img_i.name]
    elif extractor_type=='basic':
        extractor=basic.get_features
    elif extractor_type=='combine':
        combine_path=conf_dict['combine_path']
        extractor=basic.combine.build_combined(combine_path)
    else:
        extractor=getattr(basic.reduction,extractor_type)
    return extractor

if __name__ == "__main__":
    conf_path="conf/dane.cfg"
    conf_dict=utils.conf.read_config(conf_path)
    make_features(conf_dict)
    #transform_features(conf_dict)