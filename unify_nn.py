import numpy as np 
import deep.reader as reader
import basic.combine 
import utils.conf
import utils.imgs
import utils.paths.dirs as dirs
import cats
import preproc
import utils.paths.dirs as dirs
from utils.paths import Path
import basic.combine

class Unify(object):
    def __init__(self, preproc='proj',weights=None):
        self.preproc = preproc
        self.weights = weights
        
def unify(nn_paths,img_path,out_path,preproc='proj',weights=None):
    feat_dict=make_unified_feats(nn_paths,img_path,preproc,weights)
    transform_seq(img_path,out_path,feat_dict)

def make_unified_feats(nn_paths,img_path,preproc='proj',weights=None):
    combine_nn=basic.combine.build_combined(nn_paths,preproc,weights)
    imgset=utils.imgs.make_imgs(img_path,norm=True)
    feat_dir=dict([ combine_nn(img_i,True)
                    for img_i in imgset])     
    feat_dict=basic.combine.NewPathDict(feat_dir)
    return feat_dict

@dirs.apply_to_dirs
def transform_seq(in_path,out_path,extractor):
    imgs_seq= utils.imgs.make_imgs(in_path,norm=True) 
    seq=[ extractor[img_i.name] 
            for img_i in imgs_seq 
              if img_i!=None]
    txt=utils.paths.files.seq_to_string(seq)
    print(str(in_path))
    print(str(out_path))
    utils.paths.files.save_string(str(out_path)+'.txt',txt)

if __name__ == "__main__":
    #nn_paths=[ Path('../dataset2a/exp3/nn_full_100'), 
    #       Path('../dataset2a/exp3/nn_select2')]
    #nn_paths=[ Path('../dataset2a/exp3/nn_full'), 
    #           Path('../dataset2a/exp3/nn_select')]
    #nn_paths=[ Path('../dataset2a/exp3/nn_full_100'), 
    #           Path('../dataset2a/exp3/nn_select')]           
    
    #nn_paths=[ Path('../ensemble/10_nn/nn_10'), 
    #           Path('../ensemble/basic_nn/nn_basic')] 
    nn_paths=[ Path('../cross/1_set/nn'), 
               Path('../cross/1_set/nn_select')] 

    img_path=Path('../cross/full') #'../ensemble/full')
    out_path=Path('../cross/1_set/u_seq')#'../ensemble/seq')
    unify(nn_paths,img_path,out_path,weights=[1.0,1.0])