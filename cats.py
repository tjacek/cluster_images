import deep
import utils.imgs
import utils.data as data
import deep.sae as sae
import utils.conf
import utils.files as files
import utils.actions as actions
import numpy as np
import seq
import seq.dtw
from seq.to_dataset import to_dataset

def apply_sae(action_path,cls_path,out_path):
    model=files.read_object(cls_path)
    actions=utils.apply_to_dir(action_path)
    #print(len(actions)
    for action_i in actions:
        print(action_i.get_seq(model))
    seq.create_seqs(actions,out_path)

def create_sae(dir_path,conf_path,out_path):
    X,y=data.read_dataset(dir_path)
    conf_dict=utils.conf.read_config(conf_path)
    model=get_sae(conf_dict,get_init(y))
    deep.train_model_super(X,y,model) 
    files.save_object(model,out_path)

def get_sae(conf_dict,n_cats):
    ae_path=conf_dict["ae_path"] 
    ae=files.read_object(ae_path) 	
    return sae.StackedAE(ae,n_cats)

def get_init(y):
    return np.amax(y)+1
    
if __name__ == "__main__":
    path="/home/user/reps/dataset6/"
    final_path=path+"cats"
    data_path="/home/user/reps/cluster_images/dataset"
    conf_path="conf/dataset6.cfg"
    cls_path=path+"sae"
    action_path=path+"cats"
    #create_sae(data_path,conf_path,cls_path)
    apply_sae(action_path,cls_path,"dataset6.seq")
    #seq.to_dataset.to_dataset("dataset6.seq","dataset6.lb")
    #seq.dtw.wrap("dataset6.seq")