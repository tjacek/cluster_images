import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np
import utils.actions,utils.actions.read

def concat_actions(in_path1,in_path2,out_path,dataset_format='cp_dataset'):
    print(in_path1)
    action_reader=utils.actions.read.ReadActions(dataset_format,img_seq=False,as_dict=True)
    actions1=action_reader(in_path1)
    actions2=action_reader(in_path2)
    def helper_unified(key_i):
        return unified_action(actions1[key_i],actions2[key_i])
    unified_actions=[ helper_unified(key_i) 
                        for key_i in actions1]
    save_actions=utils.actions.read.SaveActions(img_actions=False)
    save_actions(unified_actions,out_path)

def unified_action(actionA,actionB):
    seqA=np.array(actionA.img_seq)
    seqB=np.array(actionB.img_seq)
    print(seqA.shape)
    print(seqB.shape)
    seqA,seqB=check_shape(seqA,seqB)
    unified_seq=np.concatenate((seqA,seqB),axis=1)
    return utils.actions.Action(actionA.name,unified_seq,
    	                        cat=actionA.cat,person=actionA.person)

def check_shape(seqA,seqB):
    a,b=seqA.shape[0],seqB.shape[0]
    print(a,b)
    if(a==b):
        return seqA,seqB
    min_dim=min(a,b)
    return seqA[:min_dim],seqB[:min_dim]

def concat_seqs(in_path1,in_path2,out_path,n_cats=20):
    for i in range(n_cats):
        in1_i=str(in_path1) + '_' + str(i)
        out_i=out_path + '_' + str(i)
        concat_actions(in1_i,in_path2,out_i)

if __name__ == "__main__":
    in_path1='../../AA_konf/all_seqs/nn'
    in_path2='../../AA_konf/untime/seq'
    out_path="../../AA_konf/full_seqs/nn"
    concat_seqs(in_path1,in_path2,out_path)