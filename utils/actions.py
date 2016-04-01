import files
#import cv2
import numpy as np
import utils.imgs as imgs
import utils.data

class Action(object):
    def __init__(self,name,frames,cat=None):
        self.name=name
        print(frames[0].shape)
        self.frames=[frame_i.reshape((1,frame_i.shape[0])) for frame_i in frames]
        self.cat=cat
        self.seq=None
    
    def as_numpy(self):
        return np.array(self.frames)

    def get_seq(self,cls):
        self.seq=[ cls.get_robust_category(frame_i) 
                         for frame_i in self.frames]
        return self.seq

    def cat_labels(self):
        return [(frame_i,self.name) for frame_i in self.frames]

    def __str__(self):
    	return self.name

    def __getitem__(self,index):
        return self.frames[index]

    def __len__(self):
        return len(self.frames)

def read_action(action_path):
    #print(action_path)
    frame_paths=files.get_files(action_path,True)
    frames= imgs.read_normalized_images(frame_paths)
    if(frames==None):
        return None
    if(len(frames)==0):
        return None
    name=files.get_name(action_path)
    cat=dir_cat(action_path,name)
    print("name: "+name)
    print("category:" + str(cat))
    print(len(frames))

    return Action(name,frames,cat)

def dir_cat(action_path,name):
    return action_path.split("/")[-2]

def name_cat(action_path,name):
    print(name)
    raw_cat=name.split("_")[0]
    raw_cat=raw_cat.replace("a","")
    return int(raw_cat)

def get_action_dataset(action_path):
    actions=utils.apply_to_dir(action_path)
    action_pairs=[action_i.cat_labels() for action_i in actions]
    all_pairs=[]
    for action_i in action_pairs:
        all_pairs+=action_i
    X,y=utils.data.pairs_to_dataset(all_pairs)
    print("y"+str(len(y)))
    return X,y