import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np
import utils.actions as actions
import utils.dirs as dirs

def agum_data(action_path,out_path):
    old_actions=actions.read_actions(action_path)
    new_actions=[flip_action(action_i)
                  for action_i in old_actions]
    all_actions=old_actions+new_actions
    dirs.make_dir(out_path)
    for action_i in all_actions:
        action_i.save(out_path)

def flip_action(action_i):
    name=action_i.name+"_h"
    cat=action_i.cat
    person=action_i.person
    img_seq=[ np.fliplr(img_i.get_orginal())
              for img_i in action_i.img_seq]
    return actions.Action(name,img_seq,cat,person)

if __name__ == "__main__":
    in_path='../dataset2a/train/'
    out_path='../dataset2a/train_agum/'
    agum_data(in_path,out_path)