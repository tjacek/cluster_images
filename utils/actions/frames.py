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

DEFAULT_DEPTH_VALUE=100

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
    def __init__(self,tau=5,scale=20,diff=False):
        self.tau=int(tau)
        self.scale=scale
        self.diff=diff

    def __call__(self,img_seq):
        if(self.diff):
            diff_seq=diff_frames(img_seq)
        else:
            diff_seq=img_seq
        #def tau_helper(diff_i):
        #    diff_i[diff_i!=0]=self.tau
        #    return diff_i
        #diff_seq=[ tau_helper(diff_i)
        #        for diff_i in diff_seq]
        n=len(diff_seq)#-1        
        #def motion_helper(img_i),img_j):
        #    motion_img=np.zeros(img_j.shape)
        #    img_i[img_j!=0]=0.0
        #    motion_img[img_i!=0]=img_i[img_i!=0]-1
        #    motion_img[img_j!=0]=img_j[img_j!=0]
        #    return utils.imgs.Image(img_i.name,motion_img,img_i.org_dim)
        def motion_helper(i):
            motion_img=np.zeros(diff_seq[0].shape)
            for t in range(self.tau):
                index=i-t
                if(index >= 0):
                    diff_t=diff_seq[index]
                    motion_img[diff_t!=0]=t
            name_i=diff_seq[i].name
            return utils.imgs.Image(name_i,motion_img)
        return [ self.scale*motion_helper(i)#diff_seq[i],diff_seq[i+1])
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
    def __init__(self,all_frames=True,clean=None,smooth_img=False):
        self.all_frames=all_frames
        self.extract_box=None
        self.clean=clean
        if(smooth_img):
            self.smooth=SmoothImg()
        else:
            self.smooth=None

    def __call__(self,img_seq):
        if(self.clean!=None):
            def clean_helper(img_i):
                img_i[:,0:self.clean]=0
                return img_i
            img_seq=[ clean_helper(img_i) for img_i in img_seq]
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
        bounded_img=extract_box_i(img_i)
        if(self.smooth!=None):
            bounded_img=self.smooth(bounded_img)
        return bounded_img

    def make_extract_box(self,nonzero):
        points=  utils.actions.bound.simple_bbox(nonzero)
        extract_box=utils.actions.bound.ExtractBox(points)
        return extract_box

class SmoothImg(object):
    def __init__(self, kern=(3,3)):
        self.kern = kern
    
    def __call__(self,raw_img):
        if(self.kern!=None):
            smooth_img=cv2.blur(raw_img,self.kern)
            smooth_img=smooth_img.astype(np.uint8)
        else:
            smooth_img=raw_img.astype(np.uint8)
        ret,binary_img=cv2.threshold(smooth_img,5,255,cv2.THRESH_BINARY)
        binary_img[binary_img!=0]=DEFAULT_DEPTH_VALUE
        return utils.imgs.Image(raw_img.name,binary_img)

def bound_local(img_seq):
    n=len(img_seq)-1
    new_seq=[utils.actions.bound.nonzero_double(img_seq[i],img_seq[i+1],True) 
                   for i in range(n)]     
    return new_seq


class ProjFrames(object):
    def __init__(self,zx=True,smooth=True):#,clean=None):
        self.zx=zx
        self.smooth=smooth

    def __call__(self,img_seq):
        z_max=upper_bound(img_seq)
        z_min=lower_bound(img_seq)
        def proj_helper(img_i):
            print(img_i.name)
            #img_i=img_i.get_orginal()
            proj_xz=self.get_clean_img( img_i,z_max,z_min)
            for (x, y), z in np.ndenumerate(img_i):
                if(z!=0):
                    #z-=z_min
                    i,j=self.get_index(x,y,z)
                    proj_xz[i][j]=DEFAULT_DEPTH_VALUE
            new_img=utils.imgs.Image(img_i.name,proj_xz)
            if(self.smooth):
                new_img=self.smooth_img(new_img)
            return new_img
        return [proj_helper(img_i) 
                   for img_i in img_seq]

    def get_clean_img(self, img_i,z_max,z_min):
        if(self.zx):
            return np.zeros( (img_i.shape[0],z_max))
        else:
            return np.zeros( (img_i.shape[1],z_max))

    def smooth_img(self,raw_img,kern=(7,7)):
        #raw_img=raw_img.astype(np.uint8)
        #smooth_img = raw_img#cv2.GaussianBlur(raw_img, (5, 5), 0)
        true_kern=np.ones(kern)
        smooth_img=remove_isol(raw_img)
        smooth_img=cv2.dilate(smooth_img, true_kern, iterations=1)
      

        #smooth_img = cv2.erode(smooth_img,(5,5),iterations = 1)
        #smooth_img=cv2.blur(raw_img,kern)
        #ret,binary_img=cv2.threshold(smooth_img,1,255,cv2.THRESH_BINARY)
        #binary_img[binary_img!=0]=DEFAULT_DEPTH_VALUE
        return utils.imgs.Image(raw_img.name,smooth_img)

    def get_index(self,x,y,z):
        z=np.floor(z)
        if(self.zx):
            return x,z
        else:
            return y,z

def remove_isol(img_i):
    kernel = np.ones((3,3),np.float32)
    kernel[1][1]=0.0
    img_i=img_i.astype(float)
    img_i[img_i!=0]=1.0
    img_i = cv2.filter2D(img_i,-1,kernel)
    img_i[ img_i<2.0]=0.0
    img_i[img_i!=0]=DEFAULT_DEPTH_VALUE
    #img_i=img_i.astype(int)

    #ret,binary_img=cv2.threshold(img_i,1,255,cv2.THRESH_BINARY)
    #binary_img[binary_img!=0]=DEFAULT_DEPTH_VALUE
    return img_i#binary_img

def upper_bound(img_seq,shift=3):
    max_z=max([np.amax(img_i)#.item()
                  for img_i in img_seq])
    max_z+=shift
    return max_z

def lower_bound(img_seq,shift=3):
    min_z=min([np.amin(img_i[np.nonzero(img_i)]).item()
                  for img_i in img_seq])
    min_z-=shift
    if(min_z<0):
        min_z=0
    return min_z