import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np 
import utils.actions.read
import utils.paths.files
from sets import Set 
import scipy.stats as st

#import pandas as pd 
import itertools

def get_global_features(in_path,out_path,dataset_format='cp_dataset'):
    read_actions=utils.actions.read.ReadActions(dataset_format,False)
    actions=read_actions(in_path)
    feat_vectors=[ extract_features(action_i)
                    for action_i in actions]
    save_global(feat_vectors,out_path)

def extract_features(action_i):
    series_i=action_i.to_series()
    data_vector=[]
    feature_extractors=[mean_of_feats,std_of_feats,skew_of_feats]#,corrl_of_feats]
    for extractor_i in feature_extractors:
        data_vector+=extractor_i(series_i)         
    return (data_vector,action_i.cat,action_i.person)	

def mean_of_feats(series_i):
    return [ np.mean(ts_j)
                  for ts_j in series_i] 

def std_of_feats(series_i,tabu=[]):
    tabu=Set(tabu)
    return [ np.std(ts_j)
                for i,ts_j in enumerate(series_i)
                  if not (i in tabu)] 

def skew_of_feats(series_i):
    return [st.skew(ts_j)
              for ts_j in series_i]

def corrl_of_feats(series_i):
    s_pairs=itertools.combinations(series_i,2)
    corl=[st.pearsonr(pair_i[0],pair_i[1])[0] 
            for pair_i in s_pairs]
    return corl

def save_global(feat_vectors,out_path):
    def extr_data(i):
    	y_i=str(feat_vectors[i][1])
    	person_i=str(feat_vectors[i][2])
        return '#'+y_i+'#'+person_i         
    x_feats=[ vector_i[0] 
              for vector_i in feat_vectors]
    feat_text= utils.paths.files.seq_to_string(x_feats,extr_data)
    utils.paths.files.save_string(out_path,feat_text)

if __name__ == "__main__":
    in_path= "../../Documents/UT/united"#'../../final_paper/UTKinect/simple/seq'
    out_path=  "../../Documents/UT/united1.txt"#'../../final_paper/UTKinect/simple/dataset.txt'
    get_global_features(in_path,out_path,dataset_format='basic_dataset')
