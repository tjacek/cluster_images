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

def transform_features(conf_dict):
    in_path=conf_dict['in_path']
    out_path=conf_dict['out_path']
    extractor=select_extractor(conf_dict)
    basic.external.transform_features(in_path,out_path,extractor) 

def make_features(conf_dict,weight=0.0):
    in_path=conf_dict['img_path']
    out_path=conf_dict['feat_path']
    extractor=select_extractor(conf_dict)
    data=imgs.make_imgs(in_path,norm=True)
    assert(type(data[0])== utils.imgs.Image )
    basic.external.external_features(out_path,data,extractor, weight)

def select_extractor(conf_dict):
    extractor_type=conf_dict['extractor']
    preproc3D=select_preproc(conf_dict)

    if(extractor_type=='deep'):
        nn_path=conf_dict['nn_path']
        nn_reader=deep.reader.NNReader(preproc3D)
        extractor=nn_reader(nn_path)
    elif extractor_type=='text':
        text_path=conf_dict['text_path']
        feat_dict=basic.external.read_external(text_path)
        extractor=lambda img_i:feat_dict[img_i.name]
    elif extractor_type=='basic':
        extractor=basic.get_features
    elif extractor_type=='combine':
        combine_path=conf_dict['combine_path']
        extractor=basic.combine.build_combined(combine_path,preproc3D)
    else:
        extractor=getattr(basic.reduction,extractor_type)
    assert(extractor!=None)
    return extractor

def select_preproc(conf_dict):
    preproc=conf_dict['preproc']
    if(preproc=='proj'):
        return deep.tools.ImgPreprocProj()
    elif preproc=='time':    
        return deep.tools.ImgPreproc2D()
    raise Exception("No preproc")

if __name__ == "__main__":
    conf_path="conf/dane4.cfg"
    conf_dict=utils.conf.read_config(conf_path)
    make_features(conf_dict)