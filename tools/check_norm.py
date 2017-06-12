import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np
import utils.actions.read

def get_norm(in_path,dataset_format='cp_dataset'):
    read_action=utils.actions.read.ReadActions(dataset_format,img_seq=True)
    actions=read_action(in_path)
    return [get_max(action_i) 
                for action_i in actions]

def norm_actions(in_path,out_path,dataset_format='cp_dataset'):
    def norm_helper(img_seq):
        max_z=get_max(img_seq)
        return [ (100.0*img_i)/max_z 
                   for img_i in img_seq ]
    utils.actions.transform_actions(in_path,out_path,
	    norm_helper,seq_transform=True,dataset_format=dataset_format)

def get_max(img_seq):
    if(type(img_seq)==utils.actions.Action):
        img_seq=img_seq.img_seq	
    max_seq=[ float(np.max(img_i))
	            for img_i in img_seq]
    return max(max_seq)

def get_min(action_i):
    def non_zero_helper(img_i):
        min_i=np.min(img_i[img_i!=0])
        return float(min_i)
    max_seq=[ non_zero_helper(img_i)
	            for img_i in action_i.img_seq]
    return max(max_seq)

if __name__ == "__main__":
    in_path= "../../exper/scale"
    out_path="../../exper/basic"
    print(get_norm(out_path))
    #norm_actions(in_path,out_path)