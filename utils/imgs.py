import numpy as np
import cv2
import paths
from dirs import dir_arg, ApplyToFiles
#from paths import str_arg

class Image(np.ndarray):
    def __new__(cls,name,input_array):
        org_dim=[input_array.shape[0],input_array.shape[1]]
        input_array=input_array.astype(float) 
        input_array=input_array.flatten()
        obj = np.asarray(input_array).view(cls) 
        obj.name=name
        obj.org_dim=org_dim
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.name = getattr(obj,'name', None)
        self.org_dim = getattr(obj,'org_dim', None)

    def get_orginal(self):
        return np.reshape(self,self.org_dim)

@dir_arg
def read_images(paths,nomalized=True):
    #print([str(path_i) for path_i in paths])
    imgs=[cv2.imread(str(p),cv2.IMREAD_GRAYSCALE) 
            for p in paths]
    imgs=[Image(path_i.get_name(),img_i) for img_i,path_i in zip(imgs,paths)
                   if img_i!=None]
    if(nomalized):
        imgs=[img_i/255.0 for img_i in imgs]
    return imgs

def save_img(full_path,img):
    img=img.get_orginal()
    img*=250.0
    img.astype(int)
    cv2.imwrite(str(full_path),img)

@ApplyToFiles(True)
@ApplyToFiles(False)
def rescale(in_path,out_path,new_dim=(60,60)):    
    img=cv2.imread(str(in_path))
    new_img=cv2.resize(img,new_dim)
    cv2.imwrite(str(out_path),new_img)