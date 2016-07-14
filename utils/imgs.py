import numpy as np
import cv2
import paths
import dirs
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

    def __str__(self):
        numbers=[ str(x_i) for x_i in self]
        return '_'.join(numbers)

    def __array_finalize__(self, obj):
        if obj is None: return
        self.name = getattr(obj,'name', None)
        self.org_dim = getattr(obj,'org_dim', None)

    def get_orginal(self):
        return np.reshape(self,self.org_dim)

@dir_arg
def read_images(paths,nomalized=True):
    return [read_img(path_i) for path_i in paths]
    
@paths.path_args
def read_img(dir_path):
    raw_img=cv2.imread(str(dir_path),cv2.IMREAD_GRAYSCALE) 
    name=dir_path.get_name()
    cat=dir_path[-2]
    img_i=Image(cat+'_'+name,raw_img)
    return img_i

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

def make_imgs(in_path,norm=False,conv=False):
    img_dirs=dirs.all_files(in_path)
    imgset=[read_img(path_i)
          for path_i in img_dirs]
    if(norm):
        imgset=[ img_i/255.0
                 for img_i in imgset]
    if(conv):
        imgset=img_forconv(imgset)
    return imgset

def img_forconv(imgset):
    imgs2D=[ img_i.get_orginal()
              for img_i in imgset]
    return np.array(imgs2D)