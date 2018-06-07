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

    def transform(self,fun, img_seq=True):
#        print(str(self))
        if(img_seq):
            img_dec=self.get_img_dec(fun)
        else:
            img_dec=fun
        new_seq=[img_dec(img_i)
                    for img_i in self.img_seq]
        new_seq=[img_i for img_i in new_seq
                         if not img_i is None]                                     
        return Action(self.name,new_seq,
                      self.cat,self.person)
    
    def feature_transform(self,fun):
        trans_features=[ fun(feature_i)
                         for feature_i in self.to_series()]
        img_seq=np.array(trans_features).T
        return Action(self.name,img_seq,self.cat,self.person)                  

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
        full_outpath=outpath.append(self.name,copy=True)
        dirs.make_dir(full_outpath)
        if(unorm):
            saved_imgs= utils.imgs.unorm(self.img_seq)
        else:
            saved_imgs= self.img_seq
        for i,img_i in enumerate(saved_imgs):   
            if(img_i!=utils.imgs.Image):
                img_i=utils.imgs.Image('img'+str(i), img_i)
            img_i.save(full_outpath,i,file_type='.png') 

    @utils.paths.path_args
    def to_text_file(self,outpath):
        full_outpath=outpath.append(self.name,copy=True)
#        print(self.img_seq)
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

    def dim(self):
        first_frame=self.img_seq[0]
        if(type(first_frame)==list):
            return len(first_frame)
        return first_frame.shape[0]

#def apply_select(in_path,out_path=None,selector=None, 
#                 dataset_format='cp_dataset',norm=False,img_seq=True):
#    read_actions=utils.actions.read.ReadActions(dataset_format,img_seq=img_seq,norm=norm)
#    actions=read_actions(in_path)
#    s_actions=raw_select(actions,selector)
#    if(out_path==None):
#        return s_actions
#    else:
#        save_actions=utils.actions.read.SaveActions(img_actions=img_seq)
#        save_actions(s_actions,out_path)

def raw_select(actions,selector=None):
    if(selector==None):
        selector=utils.selection.SelectModulo()
    if(type(selector)==int):
        selector=utils.selection.SelectModulo(selector)
    return [ action_i
               for action_i in actions
                 if selector(action_i)]

#def transform_actions(in_path,out_path,transformation,seq_transform=True,dataset_format='cp_dataset'):
#    read_actions=utils.actions.read.ReadActions(dataset_format,img_seq=True,norm=False,as_dict=False)
#    actions=read_actions(in_path)
#    print("Number of actions %d" % len(actions))
#    if(seq_transform):
#        transformed_actions=[ action_i(transformation)
#                           for action_i in actions]
#    else:
#        transformed_actions=[ action_i.transform(transformation)
#                              for action_i in actions]
#   
#    save_actions=utils.actions.read.SaveActions()
#    save_actions(transformed_actions,out_path)

def show_actions(actions):
    print([len(action_i) for action_i in actions])


class SingleFrame(object):
    def __init__(self,scale=64):
        self.scale=scale

    def __call__(self,img_i):
        new_img=img_i[0:self.scale]
        return utils.imgs.Image(img_i.name,new_img)

if __name__ == "__main__":
    full_path="../exper/full"
    basic_path="../exper/basic"
    proj_path="../exper/proj"
    #bound_frames=utils.actions.frames.ProjFrames(True,True) 
    #bound_frames=utils.actions.frames.BoundFrames(True,None,smooth_img=False) #utils.actions.frames.ProjFrames(False) 
    #rescale=utils.actions.unify.Rescale()
    time_frames= proj_set_frames()
    transform_actions(basic_path,proj_path,time_frames,seq_transform=True,dataset_format='mhad_dataset')
    
    #in_path="../cross/10_set/train_10"#preproc/unified'
    #out_path="../cross/10_set/train_select" #/preproc/train'
    #selector=utils.selection.SelectSet(['17','18'],'cat') #SelectPerson(5,True)
    #apply_select(in_path,out_path, selector,dataset_format='cp_dataset')

    