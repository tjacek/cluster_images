import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np
import utils.dirs as dirs
import utils.files as files
import utils.imgs as imgs
import utils.data
import utils.text
import utils.paths 
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
        return self.img_seq[index]

    def __len__(self):
        return len(self.img_seq)
    
    def __call__(self,func):
        return Action(self.name,func(self),
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

@utils.paths.path_args
def cp_dataset(action_dir):
    #if(type(action_dir)==str):
    #    action_dir=utils.p
    name=action_dir.get_name()
    names=name.split('_')
    if(len(names)>3):
        cat=names[0].replace('a','')
        person=int(names[1].replace('s',''))
        print(cat)
        print(person)
        print(name)
        return name,cat,person
    raise Exception("Wrong dataset format " + name +" " + str(len(names)))

def basic_dataset(action_dir):
    name=action_dir.get_name()
    cat=action_dir[-2]
    person=utils.text.get_person(name)
    return name,cat,person

FORMAT_DIR={'cp_dataset':cp_dataset,'basic_dataset':basic_dataset}

class ReadActions(object):
    def __init__(self, dataset_format,norm=False):
        if(type(dataset_format)==str):
            self.dataset_format=FORMAT_DIR[dataset_format]
        else:
            self.dataset_format=dataset_format
        self.norm=norm
        
    def __call__(self,action_path):
        action_dirs=dirs.bottom_dirs(action_path)
        actions=[self.parse_action(action_dir_i) 
                   for action_dir_i in action_dirs]
        return actions

    def parse_action(self,action_dir):
        name,cat,person=self.dataset_format(action_dir)
        img_seq=imgs.make_imgs(action_dir,norm=self.norm)
        return Action(name,img_seq,cat,person)

def select_actions(actions,action_type='odd'):
    if(action_type=='odd'):
        action_id=0
    else:
        action_id=1
    select=utils.selection.SelectModulo(action_id)
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

def apply_to_imgs(fun,actions):
    return [[fun(img_i)
              for img_i in act_i.img_seq]
                for act_i in actions]


if __name__ == "__main__":
    in_path="../dataset1/exp1/full_dataset"
    out_path="../dataset1/exp1/cats"
    read_actions=ReadActions(cp_dataset)
    actions=read_actions(in_path)
    s_actions=select_actions(actions)
    save_actions(s_actions,out_path)