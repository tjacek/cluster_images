import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np
import utils.paths.dirs as dirs
import utils.paths.files as files
import utils.imgs as imgs
import utils.data
import utils.text
import utils.paths 
import utils.selection 
import utils.actions.read
import re
from utils.actions.frames import diff_frames,motion_frames

class Action(object):
    def __init__(self,name,img_seq,cat=None,person=None):
        self.name=name
        self.img_seq=img_seq
        self.cat=cat
        self.person=person

    def __str__(self):
        return self.name

    def __getitem__(self,index):
        return self.img_seq[index]

    def __len__(self):
        return len(self.img_seq)
    
    def __call__(self,func):
        return Action(self.name,func(self.img_seq),
                       self.cat,self.person)

    def transform(self,fun):
        new_seq=[fun(img_i)
                  for img_i in self.img_seq]
        return Action(self.name,new_seq,
                      self.cat,self.person)
    
    @utils.paths.path_args
    def save(self,outpath):
        full_outpath=outpath.append(self.name,copy=True)
        dirs.make_dir(full_outpath)
        [img_i.save(full_outpath,i) 
         for i,img_i in enumerate(self.img_seq)]

def select_actions(actions,action_type='odd'):
    if(action_type=='odd'):
        action_id=1
    else:
        action_id=0
    select=utils.selection.SelectModulo(action_id)
    acts=[ action_i
           for action_i in actions
             if select(action_i.person)]
    return acts

def apply_to_imgs(fun,actions):
    return [[fun(img_i)
              for img_i in act_i.img_seq]
                for act_i in actions]

if __name__ == "__main__":
    in_path="../dataset1/preproc/depth_"
    out_path="../dataset1/preproc/bound"
    
    read_actions=utils.actions.read.ReadActions('cp_dataset',False)
    actions=read_actions(in_path)
    print( type(actions[0].img_seq[0]))
    transformed_actions=[ action_i(utils.actions.frames.bound_frames)
                           for action_i in actions]
    utils.actions.read.save_actions(transformed_actions,out_path)
    #s_actions=select_actions(actions)
    #save_actions(s_actions,out_path)