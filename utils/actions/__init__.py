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
from utils.actions.frames import diff_frames

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
    def save(self,outpath,unorm=False):
        full_outpath=outpath.append(self.name,copy=True)
        dirs.make_dir(full_outpath)
        if(unorm):
            saved_imgs= utils.imgs.unorm(self.img_seq)
        else:
            saved_imgs= self.img_seq    
        [img_i.save(full_outpath,i) 
         for i,img_i in enumerate(saved_imgs)]

def new_action(old_action,new_seq):
    return Action(old_action.name,new_seq,
                      old_action.cat,old_action.person)

def apply_select(in_path,out_path,dataset_format='cp_dataset'):
    read_actions=utils.actions.read.ReadActions(dataset_format,False)
    actions=read_actions(in_path)
    s_actions=select_actions(actions)
    utils.actions.read.save_actions(s_actions,out_path)

def transform_actions(in_path,out_path,transformation,seq_transform=True,dataset_format='cp_dataset'):
    read_actions=utils.actions.read.ReadActions(dataset_format,False)
    actions=read_actions(in_path)
    #show_actions(actions)
    if(seq_transform):
        transformed_actions=[ action_i(transformation)
                           for action_i in actions]
    else:
        transformed_actions=[ action_i.transform(transformation)
                              for action_i in actions]
    #show_actions(transformed_actions)
    utils.actions.read.save_actions(transformed_actions,out_path)

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

def show_actions(actions):
    print([len(action_i) for action_i in actions])

if __name__ == "__main__":
    #in_path="../dataset2/preproc/diff"
    #out_path="../dataset2/preproc/proj_xz"
    #bound_frames=utils.actions.frames.bound_local #(new_dim=(60,60))
    #bound_frames=utils.actions.unify.Rescale(new_dim=(60,60))#(new_dim=(60,60))
    #transform_actions(in_path,out_path,bound_frames,seq_transform=False,dataset_format='basic_dataset')
    in_path='../dataset2/exp4/full'
    out_path='../dataset2/exp4/train'
    apply_select(in_path,out_path,dataset_format='basic_dataset')

    