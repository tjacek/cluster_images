import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import seq.conds_dtw
import numpy as np
import utils.actions.read
from sets import Set 

class NNGraph(object):
    def __init__(self):
        self.names={}
        self.values={}
    
    def cliqe_reduction(self):
        removed_names=Set()
        used_set=[]
        for name_i in self.names.keys():
            if(not name_i in removed_names):
                if(self.is_clique(name_i)):
                    print("reduction")
                    removed_names.update(self.names[name_i])
                used_set.append(name_i)
        return used_set

    def is_clique(self,name_i):
        nn_names=self.names[name_i]
        nn_basic=Set(nn_names)
        for nn_i in nn_names:
            nn_i=Set(nn_i)
            if(nn_basic==nn_i):
                return False
        return True
        	
    def show(self):
        for name_i in self.names.keys():
            print(name_i)
            print(self.names[name_i])
            print(self.values[name_i])

def read_nngraph(in_path,k=3):
    dtw_pairs=utils.paths.files.read_object(in_path)
    dtw_pairs=seq.conds_dtw.DTWPairs(dtw_pairs)
    return make_nngraph(dtw_pairs,k=k)

def make_nngraph(dtw_pairs,k=2):
    nn_graph=NNGraph()
    for name_i in dtw_pairs.names(): 
        nn_keys,nn_values=dtw_pairs.get_nn(name_i,k)
        nn_graph.names[name_i]=nn_keys
        nn_graph.values[name_i]=nn_values
    return nn_graph

nn_graph=read_nngraph("../../AA_dtw/eff/corl_pairs")
print(len(nn_graph.cliqe_reduction()))