import deep
import utils.imgs
import utils.data as data
import deep.sae as sae
import utils.conf
import utils.files as files
import utils.actions as actions
import numpy as np
import seq
import seq.feats
import seq.dtw
import basic.external
from seq.to_dataset import to_dataset
import utils.actions

def apply_sae(action_path,cls_path,out_path):
    model=files.read_object(cls_path)
    actions=utils.apply_to_dir(action_path)
    #print(len(actions)
    for action_i in actions:
        print(action_i.get_seq(model))
    seq.create_seqs(actions,out_path)

def create_sae(conf_dict):
    cat_path=conf_dict['cat_path']
    X,y=data.read_dataset(cat_path)
    sae_path=conf_dict['sae_path']
    model=get_sae(conf_dict,get_init(y))
    deep.train_model_super(X,y,model) 
    files.save_object(model,sae_path)

def action_sae(conf_dict):
    action_path=conf_dict['action_path']
    sae_path=conf_dict['sae_path']
    X,y=utils.actions.get_action_dataset(action_path)
    print(y)
    model=get_sae(conf_dict,get_init(y))
    deep.train_model_super(X,y,model) 
    files.save_object(model,sae_path)

def get_sae(conf_dict,n_cats):
    ae_path=conf_dict["ae_path"] 
    ae=files.read_object(ae_path) 	
    return sae.StackedAE(ae,n_cats)

def get_init(y):
    return np.amax(y)+1
    
if __name__ == "__main__":
    conf_path="conf/dataset6.cfg"
    conf_dict=utils.conf.read_config(conf_path)
    path_dir="../dataset9/cloud"
    #action_sae(conf_dict)
    #basic.external.read_external(path_dir)
    inst=seq.feats.to_vec_seq(conf_dict)
    seq.dtw.wrap(inst)
    #create_sae(conf_dict)
    #apply_sae(action_path,cls_path,"dataset6.seq")
    #seq.to_dataset.to_string(conf_dict)#
    #to_dataset(inst,"dataset6.lb")