import numpy as np
from sets import Set
import utils.actions

def by_category(actions):
    n_cats=count_cats(actions)
    by_cat={ cat_i:[]  for cat_i in range(n_cats)}
    for action_i in actions:
        cat_i=int(action_i.cat)-1
        by_cat[cat_i].append(action_i)
    return by_cat

def count_cats(actions):
    cats=[action_i.cat for action_i in actions]	
    return len(Set(cats))

def count_feats(actions):
    return actions[0].dim()

def get_frames(actions):
    all_frames=[]
    for action_i in actions:
        all_frames+=action_i.img_seq
    return all_frames

def to_features(frames):
    frames=np.array(frames)
    frames=frames.T    
    return frames.tolist()

def new_action(old_action,new_seq):
    return utils.actions.Action(old_action.name,new_seq,
                      old_action.cat,old_action.person)

def transform_actions(actions,fun):
    return [action_i(fun) 
                for action_i in actions]