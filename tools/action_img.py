import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np 
import utils.actions
import utils.actions.read
import utils.imgs
import utils.paths.dirs

def make_action_imgs(in_path,out_path,dataset_format='cp_dataset'):
    read_actions=utils.actions.read.ReadActions(dataset_format=dataset_format,img_seq=False)
    actions=read_actions(in_path)
    action_imgs=[action_to_img(action_i)
                   for action_i in actions]
    utils.paths.dirs.make_dir(out_path)
    for action_img_i in action_imgs:
        action_img_i.save(out_path,file_type='.png')

def action_to_img(action_i):
    raw_img=np.array(action_i.img_seq)
    raw_img*=100.0
    name=action_i.name.split('.')[0]
    return utils.imgs.Image(name,raw_img)

if __name__ == "__main__":
    in_path="../../konf/hidden/seq"
    out_path="../../konf/out"
    make_action_imgs(in_path,out_path)