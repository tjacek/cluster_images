import files
#import cv2
import numpy as np
import utils.imgs as imgs

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
    #[ cv2.imread(frame_path_i) for frame_path_i in frame_paths]
    name=files.get_name(action_path)
    cat=action_path.split("/")[-2]
    print(name)
    print(cat)
    print(len(frames))
    if(frames==None):
        return None
    if(len(frames)==0):
        return None
    return Action(name,frames,cat)