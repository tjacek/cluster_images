import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np
import utils.timer
import utils.paths.dirs as dirs
import utils.imgs as imgs
import utils.paths.files as files
import utils.paths
import utils.text
import basic.reduction as redu
import basic.combine
from sets import Set
 
class ExternalFeats(object):
    def __init__(self, raw_dict,short_name=False):
        if(type(raw_dict)!=dict):
            raise Exception("Dict required")
        self.raw_dict = raw_dict
        self.short_name=short_name
        self.extract_action=lambda path_i:path_i.items[-2]

    def __getitem__(self,path_i):
        new_key=self.get_name(path_i)
        return self.raw_dict[new_key]

    def __call__(self,img_i):
        if(type(img_i)==str):
            return self.raw_dict[img_i]
        return self[img_i.name]
        #if(self.short_name):
        #    name=str(img_i.name.get_name())
        #else:
        #    name=str(img_i.name)
        #if(name in self.raw_dict):        
        #    return self.raw_dict[name]
        
        #new_key=name.split('/')[-1]
        #if(new_key in self.raw_dict):
        #    return self.raw_dict[new_key]
        #print(self.raw_dict.keys())
        #raise Exception("Key not found:"+name)

    def items(self,path=False):
        if(path):
            return [ (utils.paths.Path(key_i),value_i)
                      for key_i,value_i in self.raw_dict.items()]
        else:
            return self.raw_dict.items()
    
    def dim(self):
        return self.raw_dict.values()[0].shape[0]

    def names(self):
        return self.raw_dict.keys()

    def short_names(self):
        return { self.get_name(key_i):value_i
                           for key_i,value_i in self.raw_dict.items()}
        
    def filter_names(self,keys):
        return [key_i  
                  for key_i in keys
                    if key_i in self.raw_dict.keys()]

    def get_name(self,path_i):
        if(self.short_name==False):
            return str(path_i)
        if(type(path_i)==str):
            path_i=utils.paths.Path(path_i)
        return path_i.get_name()

    def divided_by_action(self,ordered=True):
        action_names=self.get_actions_names()
        actions={ action_i:[]
                   for action_i in action_names}
        extract_number=utils.text.ExtractNumber(use_path=True)
        for key_i,value_i in self.raw_dict.items():
            path_i=utils.paths.Path(key_i)
            action_i=self.extract_action(path_i)
            i=extract_number(path_i)
            actions[action_i].append( (i,value_i))
        if(ordered):
            actions={ action_i:order_action(value_i)
                       for action_i,value_i in actions.items()}
        return actions

    def get_actions_names(self):
        actions=Set()
        for key_i in self.raw_dict.keys():
            path_i=utils.paths.Path(key_i)
            action_i=self.extract_action(path_i)
            actions.add(action_i) 
        return list(actions)

    def save(self,out_path):
        text_dict=files.dict_to_txt(self)
        files.save_string(out_path,text_dict)

def order_action(action_list):
    n=len(action_list)
    action_set={ action_i[0]:action_i[1]
                  for action_i in action_list }
    return [action_set[i] 
              for i in range(n)]


def make_external_feat(in_path):
    text=files.read_file(in_path,lines=False)
    feat_dict=files.txt_to_dict(text)
    return feat_dict

@utils.timer.clock
def transform_features(in_path,out_path,extractor):
    feat_dict=make_external_feat(in_path)
    data=[imgs.Image(name_i,np.expand_dims(vec_i,1))
            for name_i,vec_i in feat_dict.items()]
    external_features(out_path,data,extractor,array_extr=True)

def external_features(out_path,data,extractor,weight=1.0,array_extr=False):
    if(array_extr):
        feat_dict=global_reduce(data,extractor)
    else:
        feat_dict=local_reduce(data,extractor)
    if(weight!=1.0):
        for key_i in feat_dict.keys():
            feat_dict[key_i]=weight*feat_dict[key_i]
    feat_dict=ExternalFeats(feat_dict)
    feat_dict.save(out_path)

def global_reduce(data,transform):
    names={ data_i.name:i 
                  for i,data_i in enumerate(data)}
    data_prim=transform(data)
    feat_dict={ name_i:data_prim[i]
                    for name_i,i in names.items()}
    return feat_dict#basic.combine.PathDict(feat_dict)

def local_reduce(data,transform): 
    print("%%%%%%%%%%%%%%%%%%%%%%%")
    #print(type(transform))
    #print(type(transform.preproc))
    feat_dict={ img_i.name:transform(img_i)
                for img_i in data
                  if img_i!=None}
    return feat_dict#basic.combine.PathDict(feat_dict)

def read_external(in_path,short_name=False):
    text=files.read_file(in_path,lines=False)
    feat_dict=files.txt_to_dict(text)
    get_features=ExternalFeats(feat_dict,short_name)
    return get_features

if __name__ == "__main__": 
    path_dir="../dataset1/cats"
    ae_path="../dataset1/conv.txt"
    sp_path="../wyniki/spectral.txt"
    transform_imgs(path_dir,ae_path)
    #transform_features(ae_path,sp_path)