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
import utils.actions
import re

@utils.paths.path_args
def cp_dataset(action_dir):
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

@utils.paths.path_args
def basic_dataset(action_dir):
    name=action_dir.get_name()
    cat=action_dir[-2]
    person=utils.text.get_person(name)
    return name,cat,person

@utils.paths.path_args
def cropped_dataset(action_dir):
    name=action_dir.get_name()
    names=name.split('_')
    cat=names[-3]
    person=names[-2]
    cat=cat.replace('a','')
    person=int(person.replace('s',''))
    print(cat)
    print(person)
    #cat=action_dir[-2]
    #person=utils.text.get_person(name)
    return name,cat,person

FORMAT_DIR={'cp_dataset':cp_dataset,'basic_dataset':basic_dataset,
            'cropped_dataset':cropped_dataset}

class ReadActions(object):
    def __init__(self, dataset_format,norm=False,as_dict=False):
        if(type(dataset_format)==str):
            self.dataset_format=FORMAT_DIR[dataset_format]
        else:
            self.dataset_format=dataset_format
        self.norm=norm
        self.as_dict=as_dict
        
    def __call__(self,action_path):
        action_dirs=dirs.bottom_dirs(action_path)
        #for action_i in action_dirs:
        #    print(str(action_i))
        actions=[self.parse_action(action_dir_i) 
                   for action_dir_i in action_dirs]
        if(self.as_dict):
            actions={ action_i.name:action_i
                      for action_i in actions}
        return actions

    def parse_action(self,action_dir):
        name,cat,person=self.dataset_format(action_dir)
        img_seq=imgs.make_imgs(action_dir,norm=self.norm)
        return utils.actions.Action(name,img_seq,cat,person)

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