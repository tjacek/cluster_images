import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np 
import utils.actions.read
import utils.paths.files

def simple_features(in_path,dataset_format='cp_dataset'):
    read_action=utils.actions.read.ReadActions(dataset_format=dataset_format,img_seq=False)
    actions=read_action(in_path)
    min_action=min_lenght(actions)
    print(min_action)
    def get_feats(action_i):
        pause=get_pause(action_i,min_action)
        raw_feat=[action_i[i*pause] for i in range(min_action)]
        feat_vector=np.concatenate(raw_feat) 
        return feat_vector
    dataset_str=actions_to_dataset(actions,get_feats)
    print(dataset_str)

def min_lenght(actions):
    return min([ len(action_i)
    	         for action_i in actions])

def get_pause(action_i,min_action):
    action_len=len(action_i)
    space=action_len/min_action
    print("len %d pause %d" % (action_len,space))
    return space

def actions_to_dataset(actions,fun):
    def get_feat_string(action_i): 
        num_feat_vector=fun(action_i)
        feat_str=utils.paths.files.vector_to_string(num_feat_vector)
        cat_i=str(action_i.cat)
        person_i=str(action_i.person)
        return "#".join([feat_str,cat_i,person_i])
    actions=[ get_feat_string(action_i)
	            for action_i in actions]
    return "\n".join(actions)	

if __name__ == "__main__":
    action_path= "../../final_paper/MSRaction/basic_nn/seq"#
    simple_features(action_path)