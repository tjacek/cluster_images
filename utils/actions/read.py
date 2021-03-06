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
    print(action_dir)
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
    if(len(names)>=2):
        cat= action_dir[-2]#get_basic_cat(name,action_dir)
        
        person=names[0]#utils.text.get_person(name)
        print(person)
        person=utils.text.extract_number(person)
        name=cat+'s' +str(person) +'_'+ names[1]
        print("***************")
        print(cat)
        print(person)
        print(name)
        return name,cat,person
    raise Exception("Wrong dataset format" + name +"-" + str(len(names)))        

def get_basic_cat(name,action_dir):
    if(len(action_dir)>1):
        return action_dir[-2]
    else:
        return name.split('_')[0]
         
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

@utils.paths.path_args
def mhad_dataset(action_dir):
    name=action_dir.get_name()
    desc=name.split('_')
    cat=int(desc[0].replace('a',''))
    person=int(desc[1].replace('s',''))
    return name,cat,person

FORMAT_DIR={'cp_dataset':cp_dataset,'basic_dataset':basic_dataset,
            'cropped_dataset':cropped_dataset,'mhad_dataset':mhad_dataset}

class ReadActions(object):
    def __init__(self, dataset_format,img_seq=True,norm=False,as_dict=False):
        self.dataset_format=get_dataset_format(dataset_format)
        self.norm=norm
        self.as_dict=as_dict
        self.img_seq=img_seq

    def __call__(self,action_path):
        action_dirs=self.get_dirs(action_path)
        actions=[self.parse_action(action_dir_i) 
                   for action_dir_i in action_dirs]
        if(len(actions)==0):
            raise Exception("No actions found at " + str(action_path))
        if(self.as_dict):
            def dict_helper(action_i):
                key_i=action_i.cat+'_' +action_i.name
                key_i=key_i.split('.')[0]
                return (key_i,action_i)
            actions=dict([ dict_helper(action_i)
                            for action_i in actions])
        return actions

    def parse_action(self,action_dir):
        print("sad"+str(action_dir))
        name,cat,person=self.dataset_format(action_dir)
        if(self.img_seq):
            data_seq=imgs.make_imgs(action_dir,norm=self.norm)
        else:
            data_seq=read_text_action(action_dir)
        assert len(data_seq)>0
        return utils.actions.Action(name,data_seq,cat,person)

    def get_dirs(self,action_path):
        action_dirs=dirs.bottom_dirs(action_path)
        if(not self.img_seq):
            action_paths=[]
            for action_dir_i in action_dirs:
                action_paths+=dirs.get_files(action_dir_i,dirs=False)
        else:
            action_paths=action_dirs
        if(len(action_dirs)==0):
            raise Exception("No actions in dir: " + str(action_path))
        return action_paths

class SaveActions(object):
    def __init__(self,unorm=False,img_actions=True):
        self.unorm=unorm
        self.img_actions=img_actions
        print(self.img_actions)

    @utils.paths.path_args
    def __call__(self,actions,outpath):
        dirs.make_dir(str(outpath))
        print('Save actions to ' + str(outpath))
        extr_cats=utils.data.ExtractCat(parse_cat=lambda a:a.cat)
        for action_i in actions:
            extr_cats(action_i)
        for name_i in extr_cats.names():
            print("**********")
            print(name_i)
            cat_dir_i=outpath.append(str(name_i),copy=True)
            dirs.make_dir(cat_dir_i)
        for action_i in actions:
            cat_path_i=outpath.append(str(action_i.cat),copy=True)
            if(self.img_actions and 0==1):
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
        return utils.actions.Action(name,seq_i,str(cat),str(person))

def get_dataset_format(dataset_format):
    if(type(dataset_format)==utils.paths.Path):
            dataset_format=str(dataset_format)
    if(type(dataset_format)==str):
        return FORMAT_DIR[dataset_format]
    else:
        return dataset_format

def read_text_action(input_path):
    lines=utils.paths.files.read_file(input_path,lines=True)
    return [utils.paths.files.string_to_vector(line_i)
                for line_i in lines]