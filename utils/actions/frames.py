import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np
import utils.dirs as dirs
import utils.files as files
import utils.imgs as imgs
import utils.data
import utils.text
import utils.paths 
import utils.selection 
import re

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

def diff_frames(img_seq):
    n=len(img_seq)-1
    def diff_helper(img_i,img_j):
        img_i=img_i.get_orginal()
        img_j=img_j.get_orginal()
        print(type(img_i))
        print(type(img_j))
        img_diff=img_i-img_j
        img_diff[img_diff!=0.0]=100.0
        return utils.imgs.Image(img_i.name,img_diff)
    return [ diff_helper(img_seq[i], img_seq[i+1])
             for i in range(n)]
