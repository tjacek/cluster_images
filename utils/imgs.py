import numpy as np
import utils
import cv2
import files
import basic

def read_img_as_array(action_path):
    print(action_path)
    all_files=files.get_files(action_path)
    all_files=files.append_path(action_path+"/",all_files)
    imgs=read_normalized_images(all_files)
    return np.asarray(imgs)

def read_images(files,norm_z=True):
    imgs=[cv2.imread(f,cv2.IMREAD_GRAYSCALE) for f in files]#[image.imread(f) for f in files]
    imgs=[img_i for img_i in imgs
                   if img_i!=None]
    #imgs=[img_i.reshape((1,3600)) for img_i in imgs]
    imgs=[img_i.astype(float) for img_i in imgs]
    if(norm_z):
        imgs=[img_i/255.0 for img_i in imgs]
    return imgs

def read_normalized_images(files):
    imgs=[cv2.imread(f,cv2.IMREAD_GRAYSCALE) for f in files]#[image.imread(f) for f in files]
    imgs=[img_i.flatten() for img_i in imgs
                             if img_i!=None]
    imgs=[img_i.astype(float) for img_i in imgs]
    imgs=[img_i/255.0 for img_i in imgs]
    return imgs

def read_img_dir(action_path,norm_z=True):
    print(action_path)
    all_files=files.get_files(action_path)
    return read_images(all_files,norm_z) 

def save_img(full_path,img,dim=(60,60) ):
    if(dim!=None):
        img=img.reshape(dim[0],dim[1])
    img*=250.0
    img.astype(int)
    cv2.imwrite(full_path,img)