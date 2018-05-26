import numpy as np
import utils.imgs
import cv2

class ExtractBox(object):
    def __init__(self,points):
    	if(len(points)<2):
            raise Exception("two point required ")	
        self.start_point=points[0]
        self.end_point=points[1]

    def __call__(self,img1):
        print(org_img1.shape)
    	print(self.start_point)
    	print(self.end_point)
        new_img1=org_img1[self.start_point[1]:self.end_point[1],self.start_point[0]:self.end_point[0]]
        return utils.imgs.Image(img1.name,new_img1)

def make_extract_box(action_i):
    nonzero=get_nonzero_frames(action_i.img_seq)
    points=simple_bbox(nonzero_frame)
    return ExtractBox(points)

def get_nonzero_frames(img_seq):
    first=img_seq[0]
    nonzero_frames=np.zeros(first.shape)
    for seq_i in img_seq:
        nonzero_frames[seq_i!=0]=1.0
    nonzero_frames=utils.imgs.Image(first.name,nonzero_frames,first.org_dim)
    return nonzero_frames

def simple_bbox(nonzero_frames):
    nonzero_frames*=200
    nonzero_frames=nonzero_frames.astype(np.uint8)
    thresh = cv2.threshold(nonzero_frames, 1, 255, cv2.THRESH_BINARY)[1]  
    #thresh = cv2.dilate(thresh, None, iterations=2)  
    contours, hierarchy= cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #nonzero_frames=nonzero_frames.astype(np.uint8)
    x0,y0,w,h=cv2.boundingRect(contours[0])
    return [(x0,y0),((x0+w,y0+h))]

def nonzero_double(img_x,img_y,pair=False):
    nonzero_frame=np.zeros(img_x.shape)
    nonzero_frame[img_x!=0]=1.0
    nonzero_frame[img_y!=0]=1.0
    nonzero_frame=utils.imgs.Image(img_x.name,nonzero_frame,img_x.org_dim)
    ext_box=make_extract_box(nonzero_frame)
    if(pair):
        img_i= ext_box(img_x)
        img_j= ext_box(img_y)
        final_img=np.concatenate((img_i,img_j))
        return utils.imgs.new_img(img_x,final_img)  
    else:
        return ext_box(img_x)     