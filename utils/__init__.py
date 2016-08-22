import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import cv2
import numpy as np
import utils.actions as action
import utils.imgs as imgs

def unify_list(list_of_list):
    unified_list=[]
    for list_i in list_of_list:
        unified_list+=list_i
    return unified_list

def transform_actions(in_path,out_path):
    actions=action.read_actions(in_path)
    new_actions=[action_i.transform(find_keypoints)
                 for action_i in actions ]
    action.save_actions(new_actions,out_path)

def canny_transform(img):
    raw_img=img.get_orginal()
    int_img=np.uint8(raw_img)
    canny_img=cv2.Canny(int_img,50,150)
    return imgs.Image(img.name,canny_img)

@imgs.img_arg
def find_keypoints(img_i):
    print(img_i.dtype)
    img_i=np.uint8(img_i)
    orb = cv2.ORB_create()
    keypoints = orb.detect(img_i, None)
    if(len(keypoints)>0):
         img_i=cv2.drawKeypoints(img_i,keypoints,img_i,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
         img_i = cv2.cvtColor(img_i, cv2.COLOR_BGR2GRAY)   
    return np.float64(img_i)#cv2.medianBlur(img_i,1)

if __name__ == "__main__":
    in_path='../dataset3/cats'
    out_path='../dataset3/circles'
    transform_actions(in_path,out_path) 