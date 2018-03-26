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
                    used_set.append(self.get_clique_rep(name_i))
                else:
                    used_set.append(name_i)
        return used_set

    def is_clique(self,name_i):
        nn_names=self.names[name_i]
        nn_basic=Set()
        for nn_i in nn_names:
            nn_basic.update(self.names[nn_i])
        print(len(nn_basic))
        return len(nn_basic)==(len(nn_names)+1) 
    
    def get_clique_rep(self,name_i):
        nn_names=list(self.names[name_i])
        nn_names.append(name_i)
        dists=[ sum(self.values[nn_i]) 
                    for nn_i in nn_names]
        i=np.argmax(dists)
        return nn_names[i]

    def show(self):
        for name_i in self.names.keys():
            print(name_i)
            print(self.names[name_i])
            print(self.values[name_i])

def read_nngraph(in_path,k=2):
    dtw_pairs=utils.paths.files.read_object(in_path)
    dtw_pairs=seq.conds_dtw.DTWPairs(dtw_pairs)
    return make_nngraph(dtw_pairs,k=k)

def make_nngraph(dtw_pairs,k=2):
    nn_graph=NNGraph()
    for name_i in dtw_pairs.names(): 
        nn_keys,nn_values=dtw_pairs.get_nn(name_i,k)
        print(nn_keys)
        nn_graph.names[name_i]=nn_keys
        nn_graph.values[name_i]=nn_values
    return nn_graph

def is_connected(nn_graph):
    names=nn_graph.names
    used=Set()
    def helper(names):
        unchecked=[]
        nns=[nn_graph.names[name_i] 
                for name_i in names]
        for nn_i in nns:
            if(not nn_i in used):
                used.update(nn_i)
                unchecked.append(nn_i)     
        return unchecked
    to_check=[names[0]]
    while(len(to_check)!=0):
        to_check=helper(to_check)
    return used==Set(names)



def filter_actions(in_path,out_path,actions,dataset_format='cp_dataset'):
    actions=Set(actions)
    def action_helper(action_i):
        return action_i.name in actions    
    utils.actions.apply_select(in_path,out_path,img_seq=False,
        selector=action_helper,dataset_format=dataset_format)

if __name__ == "__main__":
    nn_graph=read_nngraph("../../AA_dtw2/skew_pairs")
#nn_graph.show()
    to_filter=nn_graph.cliqe_reduction()
    print(len(to_filter))
    filter_actions("../../AA_dtw2/skew/seq","../../AA_dtw2/eff/skew",to_filter)