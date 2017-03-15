import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np 
import cv2
import utils.imgs
import utils.actions
import utils.actions.read

class UnifyActions(object):
    def __init__(self,dataset_format='cp_dataset'):
        self.read_actions=utils.actions.read.ReadActions(dataset_format,norm=False,as_dict=True)    
        self.rescale=Rescale()

    def __call__(self,in_path_x,in_path_y,out_path):
        all_actions= self.preproc_actions([in_path_x,in_path_y], [True,True])
        self.unify(all_actions)

    def append(self,in_path_x,in_path_y,out_path,norm=[False,True]):
        all_actions= self.preproc_actions([in_path_x,in_path_y], [False,True])
        self.unify(all_actions)

    def unify(self,all_actions):
        actions_x= all_actions[0]
        actions_y= all_actions[1]    
        new_actions=unify_datasets(actions_x,actions_y)
        utils.actions.read.save_actions(new_actions,out_path)

    def preproc_actions(self,paths,scaled):
        return [ self.preproc_single_action(path_i,scaled_i)
                 for path_i,scaled_i in zip(paths,scaled)]

    def preproc_single_action(self,path_i,scaled_i):
        print(str(path_i))
        actions=self.read_actions(path_i)
        print(actions.keys())
        if(scaled_i):
            actions={ name_i:action_i.transform(self.rescale)
                      for name_i,action_i in actions.items()}
        return actions

def unify_datasets(actions_x,actions_y):
    actions_names=actions_x.keys()
    return [ unify_action(actions_x[name_i],actions_y[name_i])
             for name_i in actions_names]

def unify_action(action_i,action_j):
    print(action_i.name)
    new_seq=[ unify_img(img_i,img_j)
              for img_i,img_j in zip(action_i.img_seq,action_j.img_seq,)]
    return utils.actions.new_action(action_i,new_seq)          

def unify_img(img_i,img_j):
    new_img=np.concatenate((img_i.get_orginal(),img_j.get_orginal()))
    return utils.imgs.new_img(img_i,new_img)

class Rescale(object):
    def __init__(self,new_dim=(60,60)):
        self.new_dim=new_dim

    def __call__(self,img_i):
        if(type(img_i)==utils.imgs.Image):
            img_i=img_i.get_orginal()
        new_img=cv2.resize(img_i,self.new_dim, interpolation = cv2.INTER_CUBIC)
        return utils.imgs.new_img(img_i,new_img)#,self.new_dim)

if __name__ == "__main__":
    in_path_x="../dataset2/preproc/time"
    in_path_y="../dataset2/preproc/proj_xz"
    out_path="../dataset2/preproc/unified"
    apply_unify=UnifyActions(dataset_format='basic_dataset')
    apply_unify.append(in_path_x,in_path_y,out_path,norm=[False,False])