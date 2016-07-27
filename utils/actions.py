import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np
import utils.dirs as dirs
import utils.files as files
import utils.imgs as imgs
import utils.data
import utils.text

class Action(object):
    def __init__(self,img_seq,cat=None,person=None):
        self.img_seq=img_seq
        self.cat=cat
        self.person=person
    
    def as_numpy(self):
        return np.array(self.frames)

    def cat_labels(self):
        return [(frame_i,self.cat) for frame_i in self.frames]

    #def __str__(self):
    #	return self.name

    def __getitem__(self,index):
        return self.frames[index]

    def __len__(self):
        return len(self.frames)

def read_actions(action_path):
    action_dirs=dirs.bottom_dirs(action_path)
    actions=[parse_action(action_dir_i) 
              for action_dir_i in action_dirs]
    return actions

def parse_action(action_dir):
    name=action_dir.get_name()
    cat=action_dir[-2]
    person=utils.text.get_person(name)
    img_seq=imgs.make_imgs(action_dir)
    return Action(img_seq,cat,person)


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
    return X,y

def apply_to_actions(actions,fun):
    all_actions=[]
    for action_i in actions:
        all_actions+=action_i.apply(fun)
    return all_actions

if __name__ == "__main__":
    read_actions("../dataset2/full")