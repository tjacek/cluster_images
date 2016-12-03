import numpy as np 
import deep.reader as reader
import basic.combine 
import utils.conf
import cats
import preproc

def make_feat_files(conf_paths,weights=None):
    if(weights==None):
        for conf_path_i in conf_paths:
            #preproc.make_features(conf_path_i)
            cats.easy_make_seq(conf_path_i, new_feat=True)
    else:
        for conf_path_i,weight_i in zip(conf_paths,weights):
            #preproc.make_features(conf_path_i,weight_i)
            cats.easy_make_seq(conf_path_i, new_feat=True,weight=weight_i)


def combine_feat(conf_paths,out_path):
    conf_files=[ utils.conf.read_config(conf_path_i)
                  for conf_path_i in conf_paths]
    feat_paths=[conf_file_i['feat_path'] 
                  for conf_file_i in conf_files]
    unifed_feats=basic.combine.unify_text_features_paths(feat_paths)
    print(type(unifed_feats))
    basic.external.save_features(out_path,unifed_feats)

def make_combined_seq(feat_path,img_path):
    conf_path={'feat_path':feat_path,'img_path':img_path}
    #conf_path['feat_path']=feat_path
    cats.easy_make_seq(conf_path, new_feat=False)

def unify_feats(conf_paths,weights,out_path,img_path):
    make_feat_files(conf_paths,weights)
    feat_path=out_path+'/feat.txt'
    combine_feat(conf_paths,feat_path)
    make_combined_seq(feat_path,img_path)

if __name__ == "__main__":
    conf_paths=['conf/1exp1.cfg','conf/1exp2.cfg']	
    weights=[1.0,1.0]
    out_path='../dataset1/exp3'
    img_path='../dataset1/exp3/cats'
    unify_feats(conf_paths,weights,out_path,img_path)