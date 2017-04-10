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

#def unify(paths,out_path):
#    feat_paths=[path_i.append(['feat.txt'],copy=True)
#                  for path_i in paths]
#    dirs.make_dir(out_path)
#    out_path << 'feat.txt'
#    for path_i in feat_paths:
#        print(path_i)
#    unifed_feats=basic.combine.unify_text_features_paths(feat_paths)
#    basic.external.save_features(str(out_path),unifed_feats)
#    make_combined_seq(str(out_path),'../dataset1a/exp1/full')

#def unify_feats(conf_paths,weights,out_path,img_path):
#    make_feat_files(conf_paths,weights)
#    feat_path=out_path+'/feat.txt'
#    combine_feat(conf_paths,feat_path)
#    make_combined_seq(feat_path,img_path)

#def combine():
#    feat_paths=[conf_file_i['feat_path'] 
#                  for conf_file_i in conf_files]

#def make_feat_files(conf_paths,weights=None):
#    if(weights==None):
#        for conf_path_i in conf_paths:
#            cats.easy_make_seq(conf_path_i, new_feat=True)
#    else:
#        for conf_path_i,weight_i in zip(conf_paths,weights):
#            cats.easy_make_seq(conf_path_i, new_feat=True,weight=weight_i)

#def combine_feat(conf_paths,out_path):
#    conf_files=[ utils.conf.read_config(conf_path_i)
#                  for conf_path_i in conf_paths]
#    feat_paths=[conf_file_i['feat_path'] 
#                  for conf_file_i in conf_files]
#    unifed_feats=basic.combine.unify_text_features_paths(feat_paths)
#    basic.external.save_features(out_path,unifed_feats)

#def make_combined_seq(feat_path,img_path):
#    conf_path={'feat_path':feat_path,'img_path':img_path}
#    cats.easy_make_seq(conf_path, new_feat=False)

if __name__ == "__main__":
    #nn_paths=[ Path('../dataset2a/exp3/nn_full_100'), 
    #       Path('../dataset2a/exp3/nn_select2')]
    #nn_paths=[ Path('../dataset2a/exp3/nn_full'), 
    #           Path('../dataset2a/exp3/nn_select')]
    #nn_paths=[ Path('../dataset2a/exp3/nn_full_100'), 
    #           Path('../dataset2a/exp3/nn_select')]           
    
    #nn_paths=[ Path('../ensemble/10_nn/nn_10'), 
    #           Path('../ensemble/basic_nn/nn_basic')] 
    nn_paths=[ Path('../ensemble/basic_nn/nn_basic'), 
               Path('../ensemble/10_nn/nn_10')] 

    img_path=Path('../dataset1a/AS3/full') #'../ensemble/full')
    out_path=Path('../dataset1a/AS3/seq')#'../ensemble/seq')
    unify(nn_paths,img_path,out_path,weights=[1.0,1.0])