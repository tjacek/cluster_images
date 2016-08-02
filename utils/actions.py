import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np
import utils.dirs as dirs
import utils.files as files
import utils.imgs as imgs
import utils.data
import utils.text
import utils.paths #as path
import utils.selection 

class Action(object):
    def __init__(self,name,img_seq,cat=None,person=None):
        self.name=name
        self.img_seq=img_seq
        self.cat=cat
        self.person=person

    def __str__(self):
        return self.name

    def __getitem__(self,index):
        return self.frames[index]

    def __len__(self):
        return len(self.frames)

    def transform(self,fun):
        new_seq=[fun(img_i)
                  for img_i in self.img_seq]
        return Action(self.name,new_seq,
                      self.cat,self.person)
    
    @utils.paths.path_args
    def save(self,outpath):
        full_outpath=outpath.append(self.name,copy=True)
        dirs.make_dir(full_outpath)
        [img_i.save(full_outpath) 
         for img_i in self.img_seq]

def read_actions(action_path):
    action_dirs=dirs.bottom_dirs(action_path)
    actions=[parse_action(action_dir_i) 
              for action_dir_i in action_dirs]
    return actions

def parse_action(action_dir):
    name=action_dir.get_name()
    cat=action_dir[-2]
    person=utils.text.get_person(name)
    img_seq=imgs.make_imgs(action_dir,norm=False)
    return Action(name,img_seq,cat,person)

def select_actions(actions):
    select=utils.selection.SelectModulo()
    acts=[ action_i
           for action_i in actions
             if select(action_i.person)]
    return acts

@utils.paths.path_args
def save_actions(actions,outpath):
    dirs.make_dir(outpath)
    print(dir(utils.data))
    extr_cats=utils.data.ExtractCat(parse_cat=lambda a:a.cat)
    for action_i in actions:
        extr_cats(action_i)
    for name_i in extr_cats.names():
        cat_dir_i=outpath.append(name_i,copy=True)
        dirs.make_dir(cat_dir_i)
    for action_i in actions:
        cat_path_i=outpath.append(action_i.cat,copy=True)
        action_i.save(cat_path_i)

if __name__ == "__main__":
    in_path="../dataset2a/full"
    out_path="../dataset2a/train"
    actions=read_actions(in_path)
    s_actions=select_actions(actions)
    save_actions(s_actions,out_path)