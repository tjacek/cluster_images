import utils
import utils.files as files
import seq
import numpy as np 
import feats

ALPH="?ABCDEFGHIJKLMN"

def to_string(conf_dir):
    action_path=conf_dir['action_path']
    sae_path=conf_dir['sae_path']
    sae=files.read_object(sae_path)
    actions=utils.apply_to_dir(action_path)
    actions_z=[action_i.get_seq(sae) for action_i in actions]
    str_actions=[action_to_seq(action_i) for action_i in actions] 
    for str_i in str_actions:
        print(str_i)

def action_to_seq(action):
    print(type(action))
    seq=""
    for cat_i in action.seq:
        seq+=ALPH[cat_i]
    seq+="#"+str(action.cat)
    seq+="#"+str(action.name)
    return seq

def to_dataset(instances,out_path):
    cats=[inst_i.cat for inst_i in instances]    
    cat_to_int=feats.int_cats(cats)
    cats=[cat_to_int[cat_i] for cat_i in cats]
    seqs=[inst_i.seq for inst_i in instances]
   #n_cats=len(cat_to_int.keys())+1
    instances=[]
    for cat_i,seq_i in zip(cats,seqs):
        print(seq_i)
        hist=get_histogram(seq_i,12)
        instances.append(hist_to_string(hist,cat_i))
    files.save_array(instances,out_path)

def hist_to_string(hist,cat_i):
    return files.vector_string(hist)+",#"+str(cat_i)

def seq_to_instances(str_seq,cat_names):
    raw_action=str_seq.split("#") 
    hist=get_histogram(raw_action[0])
    cat_index=utils.get_value(raw_action[1],cat_names)
    instance=files.vector_string(hist)+",#"+str(cat_index)	
    return instance

def get_histogram(raw_seq,n_cats=8):
    hist=np.zeros((n_cats,))
    for cat_index in raw_seq:
        print(cat_index)
    	hist[cat_index]+=1.0
    hist/=np.amax(hist)
    return hist    
