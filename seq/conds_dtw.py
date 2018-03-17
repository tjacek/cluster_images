import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np
import dtw
import utils.actions
import utils.actions.read
import utils.paths.files

class DTWPairs(object):
    def __init__(self):
    	self.pairs={}

    def __setitem__(self,pair,value):
        x,y=pair
        if( not (x in self.pairs)):
        	self.pairs[x]={}
        self.pairs[x][y]=value
        	
    def __getitem__(self,pair):
        return self.pairs[pair[0]][pair[1]]
    
    def get_keys(self,name):
        keys=self.pairs[name].keys()
        return [ key_i  for key_i in keys
                              if(key_i!=name)]

def save_dtw_pairs(in_path,out_path,dataset_format='cp_dataset'): 
    read_actions=utils.actions.read.ReadActions(img_seq=False,dataset_format=dataset_format)
    actions=read_actions(in_path)
    train= utils.actions.raw_select(actions,1)
    dtw_pairs=make_dtw_pairs(train)
    utils.paths.files.save_object(dtw_pairs,out_path)

def make_dtw_pairs(actions):
    dtw_pairs=DTWPairs()
    for action_i in actions:
        for action_j in actions:
            pair_ij=action_i.name,action_j.name
            value_ij=dtw.dtw_metric(action_i.img_seq,action_j.img_seq)
            dtw_pairs[pair_ij]=value_ij
            print(pair_ij)
            print(value_ij)
    return dtw_pairs

save_dtw_pairs("../../AA_dtw/max_z/seq","../../AA_dtw/max_z_pairs")
#a=utils.paths.files.read_object("../../AA_dtw/max_z_pairs")
#print(a.pairs.keys())