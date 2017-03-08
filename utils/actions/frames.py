import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np
import utils.paths.dirs as dirs
import utils.paths.files as files
import utils.imgs as imgs
import utils.data
import utils.text
import utils.paths 
import utils.selection 
import utils.actions.bound 

import re
import cv2

def motion_frames(img_seq,tau=5.0,scale=20):
    diff_seq=diff_frames(img_seq)
    diff_seq=[tau*diff_i
                for diff_i in diff_seq]
    n=len(diff_seq)-1        
    def motion_helper(img_i,img_j):
        motion_img=np.zeros(img_j.shape)
        img_i[img_j!=0]=0.0
        motion_img[img_i!=0]=img_i[img_i!=0]-1
        motion_img[img_j!=0]=img_j[img_j!=0]
        return utils.imgs.Image(img_i.name,motion_img,img_i.org_dim)
    return [ scale*motion_helper( diff_seq[i],diff_seq[i+1])
              for i in range(n) ]

def diff_frames(img_seq,threshold=0.15):
    n=len(img_seq)-1
    def diff_helper(i):
        diff_img=np.abs(img_seq[i]-img_seq[i+1])
        #print(diff_img.is_normal())
        diff_img[ diff_img>threshold]=1.0
        diff_img[ diff_img<=threshold]=0.0
        return diff_img
    return [  diff_helper(i)
              for i in range(n)]

def bound_frames(img_seq):
    nonzero= utils.actions.bound.nonzero_frames(img_seq)
    points=  utils.actions.bound.simple_bbox(nonzero)
    print(points)
    return [nonzero]

def time_frames(img_seq):
    print(type(img_seq[0]))
    n=len(img_seq)-1
    def unify_helper(img_i,img_j):
        img_i=img_i.get_orginal()
        img_j=img_j.get_orginal()

        united_img=np.array([img_i, img_j])
        new_x=united_img.shape[0]*united_img.shape[1]
        new_y=united_img.shape[2]
        img2D=united_img.reshape((new_x,new_y))
        return utils.imgs.Image(img_i.name,img2D)

    return [ unify_helper(img_seq[i], img_seq[i+1])
              for i in range(n)]

def proj_frames(img_depth):
    max_z=np.amax(img)
    img_depth*=(50.0)/max_
    img_zx=np.zeros(img_depth.shape)
    for (x_i, y_i), element in np.ndenumerate(img_depth):
        if(element!=0):
            img_y[x_i][int(element)]=50.0
    img_xy=np.zeros(img_depth.shape)
    img_xy[ img_zx!=0.0]=50.0
