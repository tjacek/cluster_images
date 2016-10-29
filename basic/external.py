import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np
import utils.timer
import utils.dirs as dirs
import utils.imgs as imgs
import utils.files as files
import utils.paths
import basic.reduction as redu

class ExternalFeats():
    def __init__(self, raw_dict,short_name=False):
        self.raw_dict = raw_dict
        self.short_name=short_name

    def __call__(self,img_i):
        if(type(img_i)==str):
            return self.raw_dict[img_i]
        
        if(self.short_name):
            name=str(img_i.name.get_name())
        else:
            name=str(img_i.name)
        if(name in self.raw_dict):        
            return self.raw_dict[name]
        else:
            raise Exception("Key not found")
            #print("Key not found")
            #print(name)
            #print(self.raw_dict.keys()[0:10])
            #return None

    def names(self):
        return self.raw_dict.keys()

    def short_names(self):
        self.raw_dict=dict([ (utils.paths.Path(key_i).get_name(),value_i)
                           for key_i,value_i in self.raw_dict.items()])
        return self
        
    def filter_names(self,keys):
        return [key_i  
                  for key_i in keys
                    if key_i in self.raw_dict.keys()]

@utils.timer.clock
def transform_features(in_path,out_path,extractor):
    text=files.read_file(in_path,lines=False)
    feat_dict=files.txt_to_dict(text)
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
    save_features(out_path,feat_dict)
    
def save_features(out_path,feat_dict):
    text_dict=files.dict_to_txt(feat_dict)
    files.save_string(out_path,text_dict)

def global_reduce(data,transform):
    names=dict([ (data_i.name,i) 
                  for i,data_i in enumerate(data)])
    data_prim=transform(data)
    feat_dict=dict([(name_i,data_prim[i])
                      for name_i,i in names.items()])
    return feat_dict

def local_reduce(data,transform): 
    print("%%%%%%%%%%%%%%%%%%%%%%%")
    feat_dict=[ (img_i.name,transform(img_i))
                for img_i in data
                  if img_i!=None]
    return dict(feat_dict)

def read_external(in_path,short_name=False):
    text='\n'.join(files.read_file(in_path))
    feat_dict=files.txt_to_dict(text)
    get_features=ExternalFeats(feat_dict,short_name)
    return get_features

if __name__ == "__main__": 
    path_dir="../dataset1/cats"
    ae_path="../dataset1/conv.txt"
    sp_path="../wyniki/spectral.txt"
    transform_imgs(path_dir,ae_path)
    #transform_features(ae_path,sp_path)