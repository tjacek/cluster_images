import numpy as np
import utils
import cv2
import files
import basic

class Image(np.ndarray):
    def __new__(cls,name,input_array):
        print(input_array.shape)
        org_dim=[input_array.shape[0],input_array.shape[1]]
        input_array=input_array.flatten()
        obj = np.asarray(input_array).view(cls) #np.ndarray.__new__(subtype, shape, dtype, buffer, offset, strides,
              #           order)
        obj.name=name
        obj.org_dim=org_dim
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.name = getattr(obj,'name', None)
        self.org_dim = getattr(obj,'org_dim', None)

    def get_orginal(self):
        return np.reshape(self,self.org_dim)

def read_img_as_array(action_path):
    print(action_path)
    all_files=files.get_files(action_path)
    all_files=files.append_path(action_path+"/",all_files)
    imgs=read_normalized_images(all_files)
    return np.asarray(imgs)

def read_images(paths,norm_z=True):
    imgs=[cv2.imread(f,cv2.IMREAD_GRAYSCALE) for f in paths]
    names=[files.get_name(path_i) for path_i in paths]
    #print(names)
    imgs=[Image(name_i,img_i) for img_i,name_i in zip(imgs,names)
                   if img_i!=None]
    #imgs=[img_i.reshape((1,3600)) for img_i in imgs]
    imgs=[img_i.astype(float) for img_i in imgs]
    if(norm_z):
        imgs=[img_i/255.0 for img_i in imgs]
    print("IMG:" +imgs[0].name)
    return imgs

def read_normalized_images(paths):
    imgs=[cv2.imread(f,cv2.IMREAD_GRAYSCALE) for f in paths]
    if(len(imgs)==0):
        return imgs
    print("OK"+ str(imgs[0].shape))
    
    names=[files.get_name(path_i) for path_i in paths]
    imgs=[Image(name_i,img_i) for img_i,name_i in zip(imgs,names)
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