import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np
import dtw
import utils.actions
import utils.actions.read
import utils.paths.files

class DTWPairs(object):
    def __init__(self,pairs={},parse=None):
        cats,parse=default_dtw_pairs(pairs,parse)
        self.pairs=pairs
        self.cats=cats
        self.parse=parse
    
    def __len__(self):
        return len(self.pairs.keys())

    def __setitem__(self,pair,value):
        x,y=pair
        if( not (x in self.pairs)):
            self.pairs[x]={}
            self.cats[x]=self.parse(x)[1]
        self.pairs[x][y]=value
        	
    def __getitem__(self,pair):
        x,y=pair
        return self.pairs[x][y]
    
    def category_separation(self,cat):
        cat_names=self.get_cats(cat)
        k=len(cat_names)
        def separation_helper(name_i):
            nn_cats=self.nn_cats(name_i,k)
            return most_cats_in_nn(cat,nn_cats)
        return sum([ int(separation_helper(name_i))
                    for name_i in cat_names])

    def without_outliners(self):
        return [key_i 
                for key_i in self.pairs.keys()
                    if self.correct_pred(key_i)]
    
    def nn_cats(self,name_i,k):
        keys,values=self.get_values(name_i)
        nn_inds=dtw.get_dist_inds(values,k)
        nn_cats=dtw.get_k_distances(keys,nn_inds)
        nn_cats=self.to_cats(nn_cats)
        return nn_cats

    def names(self):
        return self.pairs.keys()

    def get_cats(self,k):
        return [name_i for name_i in self.names()
                    if self.cats[name_i]==k]

    def to_cats(self,names):
        return [self.cats[name_i] for name_i in names]

    def correct_pred(self,correct_seq):
        pred_cat=self.nn_cats(correct_seq,k=1)[0]
        correct_cat=self.cats[correct_seq]
        return correct_cat==pred_cat
        #pred_seq=self.nearest_seq(correct_seq)
        #correct_cat=self.cats[correct_seq]
        #pred_cat=self.cats[pred_seq]
        #print(pred_cat,correct_cat)
        #return correct_cat==pred_cat

    def get_nn(self,name_i,k=1):
        keys,values=self.get_values(name_i)
        nn_inds=dtw.get_dist_inds(values,k)
        nn_keys=[keys[i] for i in nn_inds]
        nn_values=[values[i] for i in nn_inds]
        return nn_keys,nn_values

    def get_values(self,name_i):
        keys=self.get_keys(name_i)
        values=[self.pairs[name_i][key_i]
                    for key_i in keys]
        return keys,values

    def get_keys(self,name):
        keys=self.pairs[name].keys()
        return [ key_i  for key_i in keys
                              if(key_i!=name)]

def read_dtw_pairs(in_path):
    dtw_pairs=utils.paths.files.read_object(in_path)
    return DTWPairs(dtw_pairs)

def save_dtw_pairs(in_path,out_path,train=True,dataset_format='cp_dataset'): 
    read_actions=utils.actions.read.ReadActions(img_seq=False,dataset_format=dataset_format)
    actions=read_actions(in_path)
    if(train):
        train= utils.actions.raw_select(actions,1)
    else:
        train=actions
    dtw_pairs=make_dtw_pairs(train)
    utils.paths.files.save_object(dtw_pairs.pairs,out_path)

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

def default_dtw_pairs(pairs,parse):
    if(parse is None):
        parse=utils.actions.read.cp_dataset
    names=pairs.keys()
    cats={ name_i:int(parse(name_i)[1]) 
            for name_i in names} 
    return cats,parse

def all__cats_in_nn(cat,nn_cats):
    return len([ nn_cat_j
                      for nn_cat_j in nn_cats
                        if( nn_cat_j!=cat)])

def most_cats_in_nn(cat,nn_cats): 
    return dtw.most_common(nn_cats)==cat

def find_separated_cats(pairs_path,threshold=5,n_cats=20):
    dtw_pairs=utils.paths.files.read_object(pairs_path)
    dtw_pairs=DTWPairs(dtw_pairs)
    return [ i+1     
                for i in range(n_cats)
                    if(dtw_pairs.category_separation(i+1)<threshold)]

if __name__ == "__main__":
    pair_path="../../AA_dtw/eff/clique_pairs"
    #save_dtw_pairs("../../AA_dtw/eff/corl","../../AA_dtw/eff/clique_pairs",train=True)
    dtw_pairs=read_dtw_pairs(pair_path)
    print(len(dtw_pairs))
    print(len(dtw_pairs.without_outliners()))