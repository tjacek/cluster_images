import numpy as np 
import deep.reader as reader
import basic.combine 
import utils.conf
import cats

def make_feat_files(conf_paths):
    for conf_path_i in conf_paths:
        cats.easy_make_seq(conf_path_i, new_feat=True)

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

if __name__ == "__main__":
    conf_paths=['conf/exp1.cfg','conf/exp2.cfg']	
    out_path='../exp3'
    img_path='../exp3/cats'
    #make_feat_files(conf_paths)
    feat_path=out_path+'/feat.txt'
    #combine_feat(conf_paths,feat_path)
    make_combined_seq(feat_path,img_path)