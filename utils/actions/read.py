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
    if(len(names)==4):
        cat=names[0].replace('a','')
        person=int(names[1].replace('s',''))
        #print(cat)
        #print(person)
        #print(name)
        return name,cat,person
    raise Exception("Wrong dataset format " + name +" " + str(len(names)))

@utils.paths.path_args
def basic_dataset(action_dir):
    name=action_dir.get_name()
    names=name.split('_')
    if(len(names)==2):
        cat=action_dir[-2]
        person=utils.text.get_person(name)
    #name=c+name
        return name,cat,person
    raise Exception("Wrong dataset format " + name +" " + str(len(names)))        

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
    def __init__(self, dataset_format,img_seq=True,norm=False,as_dict=False):
        self.dataset_format=get_dataset_format(dataset_format)
        self.norm=norm
        self.as_dict=as_dict
        self.img_seq=img_seq

    def __call__(self,action_path):
        action_dirs=dirs.bottom_dirs(action_path)
        if(len(action_dirs)==0):
            raise Exception("No actions in dir: " + str(action_path))
        #for action_i in action_dirs:
        #    print(str(action_i))
        actions=[self.parse_action(action_dir_i) 
                   for action_dir_i in action_dirs]
        if(self.as_dict):
            actions={ action_i.cat+'_' +action_i.name:action_i
                      for action_i in actions}
        return actions

    def parse_action(self,action_dir):
        name,cat,person=self.dataset_format(action_dir)
        if(self.img_seq):
            data_seq=imgs.make_imgs(action_dir,norm=self.norm)
        else:
            data_seq=read_text_action(action_dir)
        assert len(data_seq)>0
        return utils.actions.Action(name,data_seq,cat,person)

class SaveActions(object):
    def __init__(self,unorm=False,img_actions=True):
        self.unorm=unorm
        self.img_actions=img_actions

    @utils.paths.path_args
    def __call__(self,actions,outpath):
        dirs.make_dir(outpath)
        print(type(outpath))
        extr_cats=utils.data.ExtractCat(parse_cat=lambda a:a.cat)
        for action_i in actions:
            extr_cats(action_i)
        for name_i in extr_cats.names():
            cat_dir_i=outpath.append(name_i,copy=True)
            dirs.make_dir(cat_dir_i)
        for action_i in actions:
            cat_path_i=outpath.append(action_i.cat,copy=True)
            if(self.img_actions):
                action_i.save(cat_path_i,self.unorm)
            else:
                action_i.to_text_file(cat_path_i)

class NewActions(object):
    def __init__(self,dataset_format):
        self.dataset_format=get_dataset_format(dataset_format)

    def __call__(self,actions):
        print(type(actions))
        return [ self.parse_action(name_i,data_i)
                  for name_i,data_i in actions.items()]

    def parse_action(self,name_i,seq_i):
        name,cat,person=self.dataset_format(name_i)
        return utils.actions.Action(name,str(cat),str(person),seq_i)

def get_dataset_format(dataset_format):
    if(type(dataset_format)==utils.paths.Path):
            dataset_format=str(dataset_format)
    if(type(dataset_format)==str):
        return FORMAT_DIR[dataset_format]
    else:
        return dataset_format

def read_text_action(input_path):
    lines=paths.files.read_file(input_path,lines=True)
    return [paths.files.string_to_vector(line_i)
                for line_i in lines]