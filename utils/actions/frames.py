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

class Rescale(object):
    def __init__(self,new_dim=(64,64)):
        self.new_dim=new_dim

    def __call__(self,imgs):
        if(type(imgs)==list):
            return [ self.rescale_img(img_i) for img_i in imgs]
        else:
            return self.rescale_img(imgs)

    def rescale_img(self,img_i):
        new_img=cv2.resize(img_i,self.new_dim, interpolation = cv2.INTER_CUBIC)
        return new_img#utils.imgs.new_img(img_i,new_img)

class ProjFrames(object):
    def __init__(self,zx=True,smooth=(10,10),default_depth=150.0):
        self.zx= 0 if(zx) else 1
        self.default_depth=default_depth
        self.smooth=smooth

    def __call__(self,action_i):
        print(action_i.name)
        action_seq,z_dim=prepare_seq(action_i.img_seq)
        dim_0=action_seq.shape[self.zx+1]
        dim_1=z_dim+2
        clean_img=CleanImg(dim_0,dim_1)
        def proj_helper(img_i):
            proj_i=clean_img()
            for point, z in np.ndenumerate(img_i):
                if(z!=0):
                    i=point[self.zx]
                    j=int(np.floor(z))
                    proj_i[i][j]=self.default_depth
            if(self.smooth):
                proj_i=self.smooth_img(proj_i)      
            return proj_i
        return [ proj_helper(img_i) for img_i in action_i.img_seq]

    def smooth_img(self,raw_img):
    #    se1 = cv2.getStructuringElement(cv2.MORPH_RECT, self.smooth)
    #    smooth_img = cv2.morphologyEx(raw_img, cv2.MORPH_OPEN, se1)
        true_kern=np.ones(self.smooth)
    #    smooth_img= cv2.erode(raw_img, (3,3), iterations=1)
    #    smooth_img=remove_isol(raw_img)
        smooth_img=cv2.dilate(raw_img, true_kern, iterations=1)
        return smooth_img

class CleanImg(object):
    def __init__(self,x,y):
        self.dims=(x,y)

    def __call__(self):
        return np.zeros(self.dims)
        
def time_frames(action_i):
    n=len(action_i)-1
    time_frames=[]
    for i in range(n):
        time_frames.append([action_i[i],action_i[i+1]])
    return time_frames

def prepare_seq(img_seq,z_dim=None,shift=1.0):
    if(z_dim is None):
        z_dim= float(img_seq[0].shape[0])
    action_array=np.array(img_seq)
    z_max=np.amax(action_array)+2.0*shift
    z_min=np.amin( action_array[action_array!=0])
    z_delta=z_max-z_min
    action_array[action_array!=0]+=shift
    action_array[action_array!=0]-=z_min
    action_array[action_array!=0]/=z_delta
    action_array[action_array!=0]*=z_dim
    return action_array,int(z_dim)


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

def bound_local(img_seq):
    n=len(img_seq)-1
    new_seq=[utils.actions.bound.nonzero_double(img_seq[i],img_seq[i+1],True) 
                   for i in range(n)]     
    return new_seq

def remove_isol(img_i):
    kernel = np.ones((3,3),np.float32)
    kernel[1][1]=0.0
    img_i=img_i.astype(float)
    img_i[img_i!=0]=1.0
    img_i = cv2.filter2D(img_i,-1,kernel)
    img_i[ img_i<2.0]=0.0
    img_i[img_i!=0]=DEFAULT_DEPTH_VALUE
    return img_i#binary_img