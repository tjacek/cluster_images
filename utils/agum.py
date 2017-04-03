import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np
import utils.actions as actions
import utils.actions.read
import utils.paths.dirs as dirs
import utils.paths

class FlipAction(object):
    def __init__(self):
        self.agum_name='flip'

    def __call__(self,action_i):
        new_action=action_i.transform(lambda img_i: np.fliplr(img_i))    
        new_action.name= self.agum_name+ '_' + new_action.name
        return new_action

class SubsampleAction(object):
    def __init__(self):
        self.agum_name='subsample'

    def __call__(self,action_i):
        new_seq=[ img_i
                    for i,img_i in enumerate(action_i.img_seq)
                      if (i % 2)==0]  
        new_action=action_i(lambda old_seq:new_seq)              
        new_action.name= self.agum_name+ '_' + new_action.name
        return new_action

class OutlinersAction(object):
    def __init__(self):
        self.agum_name='outliners'

    def __call__(self,action_i):
        def outliner(img_i):
            std_z=np.std(img_i)
            mean_z=np.mean(img_i)
            img_i[ (std_z+mean_z)>img_i]=0.0
            return img_i
        new_action=action_i.transform(outliner)    
        new_action.name= self.agum_name+ '_' + new_action.name
        return new_action

@utils.paths.path_args
def agum_actions(action_path,out_path,dataset_format='cp_dataset'):
    action_read=utils.actions.read.ReadActions(str(dataset_format))
    old_actions=action_read(action_path)
    flip_action=OutlinersAction()
    new_actions=[ flip_action(action_i)
                  for action_i in old_actions]
    all_actions=old_actions+new_actions
    save_actions= utils.actions.read.SaveActions()
    save_actions(all_actions,out_path)

if __name__ == "__main__":
    in_path='../ensemble/train'
    out_path='../ensemble/train_agum'
    agum_actions(in_path,out_path,'cp_dataset')