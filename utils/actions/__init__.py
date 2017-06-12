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
    
    def __call__(self,fun):
        img_dec=self.get_img_dec(fun)
        return Action(self.name,fun(self.img_seq),
                       self.cat,self.person)

    def transform(self,fun):
        img_dec=self.get_img_dec(fun)
        new_seq=[img_dec(img_i)
                  for img_i in self.img_seq]         
        return Action(self.name,new_seq,
                      self.cat,self.person)
    
    def get_img_dec(self,fun):
        def img_dec(img_i):
            new_img=fun(img_i)
            if(type(new_img)==utils.imgs.Image):
                return new_img
            else:
                return utils.imgs.Image(img_i.name,new_img)
        return img_dec

    @utils.paths.path_args
    def save(self,outpath,unorm=False):
        print(outpath)
        full_outpath=outpath.append(self.name,copy=True)
        dirs.make_dir(full_outpath)
        if(unorm):
            saved_imgs= utils.imgs.unorm(self.img_seq)
        else:
            saved_imgs= self.img_seq    
        [img_i.save(full_outpath,i,file_type='.png') 
         for i,img_i in enumerate(saved_imgs)]

    @utils.paths.path_args
    def to_text_file(self,outpath):
        full_outpath=outpath.append(self.name,copy=True)
        text_action=files.seq_to_string(self.img_seq)
        files.save_string(full_outpath,text_action)

    def to_pairs(self,index=None):
        if(index==None):
            return [ (self.cat,img_i)
                      for img_i in self.img_seq]
        else:
            return [ (self.cat,img_i,index)
                      for img_i in self.img_seq]

    def to_series(self):
        dim=len(self.img_seq[0])
        size=len(self)
        def s_helper(i):
            return [self.img_seq[j][i] 
                      for j in range(size)]
        return [ s_helper(i) 
                  for i in range(dim)]


def new_action(old_action,new_seq):
    return Action(old_action.name,new_seq,
                      old_action.cat,old_action.person)

def apply_select(in_path,out_path=None,selector=None, 
                 dataset_format='cp_dataset',norm=False):
    if(selector==None):
        selector=utils.selection.SelectModulo()
    if(type(selector)==int):
        selector=utils.selection.SelectModulo(selector)
    read_actions=utils.actions.read.ReadActions(dataset_format,norm=norm)
    actions=read_actions(in_path)
    s_actions=raw_select(actions,selector)
    if(out_path==None):
        return s_actions
    else:
        save_actions=utils.actions.read.SaveActions()
        save_actions(s_actions,out_path)

def raw_select(actions,selector):
    return [ action_i
               for action_i in actions
                 if selector(action_i)]

def transform_actions(in_path,out_path,transformation,seq_transform=True,dataset_format='cp_dataset'):
    read_actions=utils.actions.read.ReadActions(dataset_format,img_seq=True,norm=False,as_dict=False)
    actions=read_actions(in_path)
    print("Number of actions %d" % len(actions))
    if(seq_transform):
        transformed_actions=[ action_i(transformation)
                           for action_i in actions]
    else:
        transformed_actions=[ action_i.transform(transformation)
                              for action_i in actions]
   
    save_actions=utils.actions.read.SaveActions()
    save_actions(transformed_actions,out_path)

def show_actions(actions):
    print([len(action_i) for action_i in actions])

if __name__ == "__main__":
    in_path="../../exper/scale"
    out_path="../../exper/time"
    #bound_frames=utils.actions.frames.ProjFrames(True,True) 
    #bound_frames=utils.actions.frames.BoundFrames(True,None,smooth_img=False) #utils.actions.frames.ProjFrames(False) 
    #rescale=utils.actions.unify.Rescale()
    time_frames=utils.actions.frames.TimeFrames() 
    transform_actions(in_path,out_path,time_frames,seq_transform=True,dataset_format='cp_dataset')
    
    #in_path="../cross/10_set/train_10"#preproc/unified'
    #out_path="../cross/10_set/train_select" #/preproc/train'
    #selector=utils.selection.SelectSet(['17','18'],'cat') #SelectPerson(5,True)
    #apply_select(in_path,out_path, selector,dataset_format='cp_dataset')

    