import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import cv2
import numpy as np
import utils.actions as action
import utils.imgs as imgs

def transform_actions(in_path,out_path):
    actions=action.read_actions(in_path)
    new_actions=[action_i.transform(canny_transform)
                 for action_i in actions ]
    action.save_actions(new_actions,out_path)

def canny_transform(img):
    raw_img=img.get_orginal()
    int_img=np.uint8(raw_img)
    canny_img=cv2.Canny(int_img,50,150)
    return imgs.Image(img.name,canny_img)

if __name__ == "__main__":
    in_path='../dataset3/cats'
    out_path='../dataset3/edges'
    transform_actions(in_path,out_path) 