import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np
import copy
import utils.actions,utils.actions.read
import utils.selection

def bagged_dataset(in_path,out_path,dataset_format='cp_dataset'):
    read_actions=utils.actions.read.ReadActions(dataset_format,img_seq=False)
    actions=read_actions(in_path)
    new_actions=bagged_actions(actions)
    save_actions=utils.actions.read.SaveActions(dataset_format,img_actions=False)
    save_actions(new_actions,out_path)

def bagged_dataset_select(in_path,out_path,dataset_format='cp_dataset'):
    read_actions=utils.actions.read.ReadActions(dataset_format,img_seq=False)
    actions=read_actions(in_path)
    
    train,test=split_dataset(actions)
    
    bag_actions=bagged_actions(train)
    all_actions=bag_actions+test
    save_actions=utils.actions.read.SaveActions(dataset_format,img_actions=False)
    save_actions(all_actions,out_path)

def bagged_actions(actions):
    n_actions=len(actions)
    def rand(i):
        rand_index=np.random.randint(0,n_actions)
        rand_action= copy.copy(actions[rand_index])
        str_id=str(i)+'.'
        new_name=rand_action.name.replace('.',str_id)
        rand_action.name=new_name
        return rand_action
    new_actions=[ rand(i)
                  for i in range(n_actions)]
    return new_actions

def split_dataset(actions):
    odd=utils.selection.SelectModulo(0)
    even=utils.selection.SelectModulo(1)
    train=utils.actions.raw_select(actions,even)  
    test=utils.actions.raw_select(actions,odd)  
    return train,test

if __name__ == "__main__": 
    in_path='../bagging/basic_nn/seq2'
    out_path='../bagging/bag3/seq'
    bagged_dataset_select(in_path,out_path,dataset_format='cp_dataset')