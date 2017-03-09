import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np 
import cv2
import utils.imgs
import utils.actions
import utils.actions.read

def apply_unify(in_path_x,in_path_y,out_path,dataset_format='cp_dataset'):
    read_actions=utils.actions.read.ReadActions(dataset_format,norm=False,as_dict=True)	
    actions_x=read_actions(in_path_x)
    actions_y=read_actions(in_path_y)
    new_actions=unify_datasets(actions_x,actions_y)
    utils.actions.read.save_actions(new_actions,out_path)

def unify_datasets(actions_x,actions_y):
    actions_names=actions_x.keys()
    return [ unify_action(actions_x[name_i],actions_y[name_i])
             for name_i in actions_names]

def unify_action(action_i,action_j):
    action_i=action_i.transform(rescale)
    action_j=action_j.transform(rescale)
    print(action_i.name)
    new_seq=[ unify_img(img_i,img_j)
              for img_i,img_j in zip(action_i.img_seq,action_j.img_seq,)]
    return utils.actions.new_action(action_i,new_seq)          

def unify_img(img_i,img_j):
    new_img=np.concatenate((img_i.get_orginal(),img_j.get_orginal()))
    return utils.imgs.new_img(img_i,new_img)

def rescale(img_i,new_dim=(60,60)):
    #print(img_i.name)
    new_img=cv2.resize(img_i.get_orginal(),new_dim, interpolation = cv2.INTER_CUBIC)
    return utils.imgs.new_img(img_i,new_img,new_dim)

if __name__ == "__main__":
    in_path_x="../dataset1/preproc/diff_xy_"
    in_path_y="../dataset1/preproc/diff_xz_"
    out_path="../dataset1/preproc/unified"
    apply_unify(in_path_x,in_path_y,out_path)