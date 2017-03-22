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

@utils.paths.path_args
def agum_actions(action_path,out_path,dataset_format='cp_dataset'):
    action_read=utils.actions.read.ReadActions(str(dataset_format))
    old_actions=action_read(action_path)
    flip_action=FlipAction()
    new_actions=[ flip_action(action_i)
                  for action_i in old_actions]
    all_actions=old_actions+new_actions
    utils.actions.read.save_actions(all_actions,out_path)

#def agum_data(action_path,out_path):
#    old_actions=actions.read_actions(action_path)
#    new_actions=[flip_action(action_i)
#                  for action_i in old_actions]
#    all_actions=old_actions+new_actions
#    dirs.make_dir(out_path)
#    for action_i in all_actions:
#        path_i=utils.paths.Path(out_path)
#        path_i.add(action_i.cat)
#        dirs.make_dir(path_i)
#        action_i.save(str(path_i))

#def subsample_action(action_i):
#    name=action_i.name+"_h"
#    cat=action_i.cat
#    person=action_i.person
#    img_seq=[ img_i
#              for i,img_i in enumerate(action_i.img_seq)
#                if (i%2)==0]
#    return actions.Action(name,img_seq,cat,person)

if __name__ == "__main__":
    in_path='../dataset2a/exp3/train'
    out_path='../dataset2a/exp3/train_agum'
    agum_actions(in_path,out_path,'basic_dataset')
    #in_path='../dataset2a/train/'
    #out_path='../dataset2a/train_agum/'
    #agum_data(in_path,out_path)