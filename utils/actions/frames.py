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
from utils.actions.unify import Rescale 
import re
import cv2

DEFAULT_DEPTH_VALUE=100.0

class TimeFrames(object):
    def __init__(self, new_dim=None):
        self.new_dim=new_dim
        self.rescale=Rescale(new_dim)

    def __call__(self,img_seq):
        print(type(img_seq[0]))
        n=len(img_seq)-1
        def unify_helper(img_i,img_j):
            img_i=img_i.get_orginal()
            img_j=img_j.get_orginal()
            if(self.new_dim!=None):
                img_i=self.rescale(img_i)
                img_j=self.rescale(img_j)
            united_img=np.array([img_i, img_j])
            new_x=united_img.shape[0]*united_img.shape[1]
            new_y=united_img.shape[2]
            img2D=united_img.reshape((new_x,new_y))
            return utils.imgs.Image(img_i.name,img2D)
        print(len(img_seq))
        new_seq=[ unify_helper(img_seq[i], img_seq[i+1])
                 for i in range(n)]
        print(len(new_seq))
        return new_seq

class MotionFrames(object):
    def __init__(self,tau=5.0,scale=20):
        self.tau=tau
        self.scale=scale

    def __call__(self,img_seq):
        diff_seq=diff_frames(img_seq)
        diff_seq=[self.tau*diff_i
                for diff_i in diff_seq]
        n=len(diff_seq)-1        
        def motion_helper(img_i,img_j):
            motion_img=np.zeros(img_j.shape)
            img_i[img_j!=0]=0.0
            motion_img[img_i!=0]=img_i[img_i!=0]-1
            motion_img[img_j!=0]=img_j[img_j!=0]
            return utils.imgs.Image(img_i.name,motion_img,img_i.org_dim)
        return [ self.scale*motion_helper( diff_seq[i],diff_seq[i+1])
                   for i in range(n) ]

def diff_frames(img_seq,threshold=0.1):
    n=len(img_seq)-1
    def diff_helper(i):

        diff_img=np.abs(img_seq[i]-img_seq[i+1])
        #print(diff_img.is_normal())
        diff_img[ diff_img>=threshold]=DEFAULT_DEPTH_VALUE
        diff_img[ diff_img<threshold]=0.0
        return diff_img
    return [  diff_helper(i)
              for i in range(n)]

class BoundFrames(object):
    def __init__(self,all_frames=True):
        self.all_frames=all_frames
        self.extract_box=None

    def __call__(self,img_seq):
        if(self.all_frames):
            nonzero= utils.actions.bound.nonzero_frames(img_seq)
            self.extract_box=self.make_extract_box(nonzero)
        return [ self.get_box(img_i) 
                  for img_i in img_seq]
        
    def get_box(self,img_i):
        if(self.extract_box==None):
            extract_box_i=self.make_extract_box(img_i)
        else:
            extract_box_i=self.extract_box
        return extract_box_i(img_i)

    def make_extract_box(self,nonzero):
        points=  utils.actions.bound.simple_bbox(nonzero)
        extract_box=utils.actions.bound.ExtractBox(points)
        return extract_box

def bound_frames(img_seq):
    #print(type(img_seq))
    #print([img_i.get_orginal() for img_i in img_seq])
    nonzero= utils.actions.bound.nonzero_frames(img_seq)
    points=  utils.actions.bound.simple_bbox(nonzero)
    extract_box=utils.actions.bound.ExtractBox(points)
    return [ extract_box(img_i)
              for img_i in img_seq]

class BoundLocal(object):
    def __init__(self,new_dim=None):
        if(new_dim!=None):
            self.rescale=Rescale(new_dim)
        else:
            self.rescale=None

    def __call__(self,img_seq):
        n=len(img_seq)-1
        
        new_seq=[bound_frames([img]) 
                   for i in range(n)]
        #new_seq=[utils.actions.bound.nonzero_double(img_seq[i],img_seq[i+1],True) 
        #           for i in range(n)]
        if(self.rescale!=None):
            new_seq=[self.rescale(img_i) 
                       for img_i in new_seq]
        return new_seq


class ProjFrames(object):
    def __init__(self,zx=True):
        self.zx=zx

    def __call__(self,img_seq):
        z_max=max_frames(img_seq)+2
        def proj_helper(img_i):
            print(img_i.name)
            img_i=img_i.get_orginal()
            proj_xz=self.get_clean_img( img_i,z_max)
            for (x, y), z in np.ndenumerate(img_i):
                i,j=self.get_index(x,y,z)
                proj_xz[i][j]=DEFAULT_DEPTH_VALUE
            return utils.imgs.Image(img_i.name,proj_xz)
        return [proj_helper(img_i) 
                   for img_i in img_seq]

    def get_clean_img(self, img_i,z_max):
        if(self.zx):
            return np.zeros( (img_i.shape[0],z_max))
        else:
            return np.zeros( (img_i.shape[1],z_max,))

    def get_index(self,x,y,z):
        z=np.floor(z)
        if(self.zx):
            return x,z
        else:
            return y,z

def max_frames(img_seq):
    return max([np.amax(img_i)
                  for img_i in img_seq])