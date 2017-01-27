import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np 
import utils.dirs
import utils.imgs
import utils.paths
import deep.reader
import basic.external

class CombinedFeatures(object):
    def __init__(self, extractors):
        self.extractors = extractors

    def __call__(self,img_i):
        feats=[extr_i(img_i) 
    	        for extr_i in self.extractors]
    	feats=np.concatenate(feats)
    	return feats

def build_combined(in_path,preproc=None):
    all_paths=utils.dirs.all_files(in_path)
    nn_reader=deep.reader.NNReader()
    extractors=[nn_reader(nn_path_i,preproc,drop_p=0.0)
                  for nn_path_i in all_paths]
    return CombinedFeatures(extractors)

def unify_text_features(in_path):
    all_paths=utils.dirs.all_files(in_path)
    unify_text_features_paths(all_paths)

class PathDict(object):
    def __init__(self,raw_dict=None):
        if(raw_dict==None):
            self.raw_dict={}
        elif( type(raw_dict)==list):
            self.raw_dict=dict(raw_dict)
        else:    
            self.raw_dict=raw_dict

    def __setitem__(self, key, item):
        self.raw_dict[ self.name(key)]=item
        self.raw_dict[key] = item

    def __getitem__(self, key):
        if(key in self.raw_dict[key]):
            return self.raw_dict[key]
        new_key=key.split('/')[-1]
        return self.raw_dict[new_key]
    
    def name(self,key):
        return key.split('/')[-1]

    def items(self):
        return self.raw_dict.items()
    
    def keys(self):
        return self.raw_dict.keys()

def unify_text_features_paths(all_paths):
    dict_features=[basic.external.read_external(path_i).short_names()
                        for path_i in all_paths]
    img_names=dict_features[0].names()
    for dict_i in dict_features:
        img_names=dict_i.filter_names(img_names)
    unifed_feats={}#PathDict()
    for key_i in img_names:
        feats_i=[dict_j.raw_dict[key_i]
                 for dict_j in dict_features]
        feats_i=np.concatenate(feats_i,axis=0)
        print(feats_i.shape)
        unifed_feats[key_i]=feats_i
    return unifed_feats

if __name__ == "__main__":
    in_path="../dane3/feats"
    out_path="../dane3/out_path.txt"
    unifed_feats=unify_text_features(in_path)
    basic.external.save_features(out_path,unifed_feats)
