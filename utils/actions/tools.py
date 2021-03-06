import numpy as np
from sets import Set
import utils.actions
import utils.selection 

class ActionTransform(object):
    def __init__(self,transform_type='img',in_seq=True,
                      out_seq=False, dataset_format='basic_dataset'):
        self.transform_type=transform_type
        self.out_seq=out_seq
        self.read=utils.actions.read.ReadActions(dataset_format=dataset_format,img_seq=in_seq)
        self.save=utils.actions.read.SaveActions(img_actions=out_seq)

    def __call__(self,in_path,out_path,transformation):
        actions=self.read(in_path)
        print("Number of actions %d" % len(actions))
        print(actions[0].img_seq[0].shape)
        if(self.transform_type=='img_seg'):
            transformed_actions=[ action_i(transformation)
                           for action_i in actions]
        elif(self.transform_type=='img'):
            transformed_actions=[ action_i.transform(transformation,img_seq=self.out_seq)
                              for action_i in actions]
        else:
            transformed_actions=[ transformation(action_i)
                                    for action_i in actions]
        self.save(transformed_actions,out_path)

class ActionSelection(object):
    def __init__(self,in_seq=True,out_seq=False, dataset_format='cp_dataset'):
        self.read=utils.actions.read.ReadActions(dataset_format=dataset_format,img_seq=in_seq)
        self.save=utils.actions.read.SaveActions(img_actions=out_seq)
    
    def __call__(self, in_path,out_path,selector):
        actions=self.read(in_path)
        if(type(selector)==int):
            selector=utils.selection.SelectModulo(selector)
        s_actions=[ action_i 
                    for action_i in actions
                        if(selector(action_i))]
        self.save(s_actions,out_path)

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
