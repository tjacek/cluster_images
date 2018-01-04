import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np
import utils.actions,utils.actions.read

def concat_actions(in_path1,in_path2,out_path,dataset_format='cp_dataset'):
    action_reader=utils.actions.read.ReadActions(dataset_format,img_seq=False,as_dict=True)
    actions1=action_reader(in_path1)
    actions2=action_reader(in_path2)
    def helper_unified(key_i):
        actionA=actions1[key_i]
        actionB=actions2[key_i]
        return unified_action(actionA,actionB)
    unified_actions=[ helper_unified(key_i) 
                        for key_i in actions1]
    save_actions=utils.actions.read.SaveActions(img_actions=False)
    save_actions(unified_actions,out_path)

def unified_action(actionA,actionB):
    seqA=np.array(actionA.img_seq)
    seqB=np.array(actionB.img_seq)
    unified_seq=np.concatenate((seqA,seqB),axis=1)
    return utils.actions.Action(actionA.name,unified_seq,
    	                        cat=actionA.cat,person=actionA.person)

if __name__ == "__main__":
    in_path1= '../../AArtyk/simple/corl/seq'
    in_path2= '../../AArtyk/simple/skew/seq'
    out_path= '../../AArtyk/simple/conc'
    concat_actions(in_path1,in_path2,out_path)