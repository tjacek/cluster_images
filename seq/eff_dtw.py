import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np
import utils.actions
import utils.actions.read
from sets import Set
import seq.dtw

def select_cats(in_path,out_path,cats,dataset_format='cp_dataset'):
    cats=Set(cats)
    def cat_helper(action_i):
        return int(action_i.cat) in cats	
    actions=utils.actions.apply_select(in_path,out_path,img_seq=False,
    	selector=cat_helper,dataset_format=dataset_format)

def eff_feats(data_path,train_path,out_path,dataset_format='cp_dataset'):
    all_actions=get_actions(data_path,train=False,dataset_format=dataset_format)
    train_actions=get_actions(train_path,train=True,dataset_format=dataset_format)
    def feat_helper(action_i):
    	print(action_i.name)
        feats=[seq.dtw.dtw_metric(action_i.img_seq,action_j.img_seq)
                for action_j in train_actions]
        return np.array(feats)
    dtw_feats=[ feat_helper(action_i) for action_i in all_actions]
    def extr_data(i):
        y_i=str(all_actions[i].cat)
        person_i=str(all_actions[i].person)
        return '#'+y_i+'#'+person_i         
    feat_text= utils.paths.files.seq_to_string(dtw_feats,extr_data)
    utils.paths.files.save_string(out_path,feat_text)

def get_actions(in_path,train=False,dataset_format='cp_dataset'):
    read_actions=utils.actions.read.ReadActions(img_seq=False,dataset_format=dataset_format)
    actions=read_actions(in_path)
    if(train):
        return utils.actions.raw_select(actions,1)
    return actions
    
if __name__ == "__main__":
    eff_feats("../../AA_dtw/skew/seq","../../AA_dtw/eff/skew/clique","../../AA_dtw/eff/clique_feats.txt")
  