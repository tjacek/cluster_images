import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np
import dtw
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

def save_dtw_pairs(in_path,out_path,dataset_format='cp_dataset'): 
    read_actions=utils.actions.read.ReadActions(img_seq=False,dataset_format=dataset_format)
    actions=read_actions(in_path)
    dtw_pairs=make_dtw_pairs(actions)
    utils.paths.files.save_object(dtw_pairs,out_path)

def make_dtw_pairs(actions):
    dtw_pairs=DTWPairs()
    for action_i,action_j in zip(actions,actions):
        pair_ij=action_i.name,action_j.name
        value_ij=dtw.dtw_metric(action_i.img_seq,action_j.img_seq)
        dtw_pairs[pair_ij]=value_ij
        print(pair_ij)
    return dtw_pairs

save_dtw_pairs("../../AA_dtw/max_z/seq","../../AA_dtw/test")