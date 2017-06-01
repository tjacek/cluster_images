import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np 
import utils.actions.read
import utils.paths.files
from sets import Set 

def get_global_features(in_path,out_path,dataset_format='cp_dataset'):
    read_actions=utils.actions.read.ReadActions(dataset_format,False)
    actions=read_actions(in_path)
    feat_vectors=[ extract_features(action_i)
                    for action_i in actions]
    save_global(feat_vectors,out_path)

def extract_features(action_i):
    series_i=action_i.to_series()
    mean_vector=mean_of_feats(series_i)
    sd_vector=std_of_feats(series_i)
    data_vector=mean_vector+sd_vector            
    return (data_vector,action_i.cat,action_i.person)	

def mean_of_feats(series_i):
    return [ np.mean(ts_j)
                  for ts_j in series_i] 

def std_of_feats(series_i,tabu=[]):
    tabu=Set(tabu)
    return [ np.std(ts_j)
                for i,ts_j in enumerate(series_i)
                  if not (i in tabu)] 

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
    in_path= '../../ultimate3/simple/seq'
    out_path= '../../ultimate3/simple/dataset.txt'
    get_global_features(in_path,out_path)
