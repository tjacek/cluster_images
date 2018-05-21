import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np
import utils.actions,utils.actions.read

def add_actions_concat(in_path1,in_path2, out_path,n_cats=20,dataset_format='cp_dataset'):
    for i in range(n_cats):
        in_i=in_path2+'_'+str(i)
        out_i=out_path+'_'+str(i)
        pair_actions_concat(in_path1,in_i,out_i,dataset_format)

def all_actions_concat(in_path,out_path,n_cats=20,dataset_format='cp_dataset'):
    paths=[in_path+'_'+str(i)  for i in range(n_cats)]  
    actions_concat(paths,out_path,dataset_format)

def pair_actions_concat(in_path1,in_path2,out_path,dataset_format='cp_dataset'):
    actions_concat([in_path1,in_path2],out_path,dataset_format)

def actions_concat(paths,out_path,dataset_format='cp_dataset'):
    action_reader=utils.actions.read.ReadActions(dataset_format,img_seq=False,as_dict=True)
    all_actions=[action_reader(path_i) for path_i in paths]
    by_name=get_actions_by_name(all_actions)
    unified_actions=[ unifiy_actions(actions_i) for actions_i in by_name]
    save_actions=utils.actions.read.SaveActions(img_actions=False)
    save_actions(unified_actions,out_path)

def get_actions_by_name(all_actions):
    action_names=all_actions[0].keys()
    return [[actions_j[name_i]
                for actions_j in all_actions]
                    for name_i in action_names]

def unifiy_actions(actions):
    action_0=actions[0]
    seqs=[np.array(action_i.img_seq) for action_i in actions]
    seqs=check_shape(seqs)
    unified_seq=np.concatenate(seqs,axis=1)
    return utils.actions.Action(action_0.name,unified_seq,
                               cat=action_0.cat,person=action_0.person)

def check_shape(seqs):
    dims=[seq_i.shape[0]  for seq_i in seqs]
    min_dim=min(dims)
    if(min_dim==max(dims)):
        return seqs
    return [seq_i[:min_dim] for seq_i in seqs]

if __name__ == "__main__":
    in_path2="../../Documents/EE/simple"
    in_path1="../../Documents/EE/seq3"
    out_path="../../Documents/EE/united"
#    add_actions_concat(in_path1,in_path2,out_path)
#    all_actions_concat(in_path,full_path)
    pair_actions_concat(in_path1,in_path2,out_path,dataset_format='basic_dataset')
#    in_path2='../../AA_disk2/basic_norm/'
#    in_path1="../../AA_disk3/clust_seqs/nn"
#    out_path="../../AA_disk3/united_seqs/nn"
#    concat_actions(in_path1,in_path2,out_path)
#    concat_seqs(in_path1,in_path2,out_path)