import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import scipy.io as sio
import numpy as np

import utils.imgs
import utils.paths.dirs
import utils.actions,utils.actions.read


def to_actions(in_path,out_path):
    paths,actions=read_mat(in_path)
    actions=[  to_action(path_i,array_i)
               for path_i,array_i in zip(paths,actions)]
    save_actions=utils.actions.read.SaveActions()
    save_actions(actions,out_path)

def read_mat(in_path):
    paths=utils.paths.dirs.all_files(in_path,append_path=True)
    def load(path_i):
        print(str(path_i))
        mat_file=sio.loadmat(str(path_i))
        depth_array=mat_file['d_depth'].astype(float)
        return norm(depth_array)
    actions=[ load(path_i) for path_i in paths]
    print(actions[0].dtype)
    return (paths,actions)

def to_action(path_i,np_array):
    img_seq=get_img_seq(np_array)
    name,cat,person=parse_action(path_i)
    return utils.actions.Action(name,img_seq,cat,person)
    
def parse_action(path_i):
    name=path_i.get_name()
    desc=name.split('_')
    cat=int(desc[0].replace('a',''))
    person=int(desc[1].replace('s',''))
    return name,cat,person
  
def get_img_seq(np_array,postfix='.jpg'):
    size=np_array.shape[2]
    def seq_helper(i):
        name="img"+str(i)+postfix
        return utils.imgs.Image(name,np_array[:,:,i])
    img_seq=[ seq_helper(i)
              for i in range(size)]
    return img_seq

def norm(np_array,max_value=100.0):
    a_max=np.amax(np_array)
    a_min=np.amin(np_array[np_array!=0])
    np_array[np_array!=0]-= (a_min-1.0)    
    scale=(a_max-a_min)
    np_array/=scale
    np_array*=max_value    
    return np_array


in_path='../../AArtyk2/raw'
out_path='../../AArtyk2/depthols/'

to_actions(in_path,out_path)