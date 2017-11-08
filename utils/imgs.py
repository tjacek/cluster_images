import numpy as np
import cv2
import paths
import paths.dirs
from paths.dirs import dir_arg, ApplyToFiles

class Image(np.ndarray):
    def __new__(cls,name,input_array,org_dim=None):
        if(org_dim==None):
            org_dim=[input_array.shape[0],input_array.shape[1]]
        input_array=input_array.astype(float) 
        #input_array=input_array.flatten()
        obj = np.asarray(input_array).view(cls)
        obj.name=paths.Path(name)
        obj.org_dim=org_dim
        return obj

#    def __str__(self):
#        return str(self.name)

    def __array_finalize__(self, obj):
        if obj is None: return
        self.name = getattr(obj,'name', None)
        self.org_dim = getattr(obj,'org_dim', None)

    def get_orginal(self):
        return np.reshape(self,self.org_dim)

    def save(self,out_path,i=None,unorm=False,file_type='.jpg'):
        if(type(out_path)==str):
            out_path=paths.Path(out_path)
        if(i!=None):
            filename= 'img' +str(i)+file_type #self.name
        else:
            filename= get_name(self.name) + file_type
        full_name=out_path.append(filename,copy=True)
        img2D=self.get_orginal()
        print(str(full_name))
        if(unorm):
            save_img(full_name,img2D)
        else:
            cv2.imwrite(str(full_name),img2D)

    def is_normal(self):
        return not (np.amax(self)>1.0)

def save_imgset(out_path,imgset):
    #print(str(out_path))
    paths.dirs.make_dir(out_path)
    for i,img_i in enumerate(imgset):
        print(str(img_i.name))
        img_i.save(out_path,unorm=True)

def get_name(path_i):
    if(type(path_i)==str):
        path_i=paths.Path(path_i)
        return path_i.last(2)#[-2]
    return path_i.get_name()

def new_img(old_img,new_img,org_dim=None):
    if(org_dim!=None):
        return  Image(old_img.name,new_img,org_dim)
    return Image(old_img.name,new_img)

def img_arg(func):
    def inner_func(img_i):
        org_i=img_i.get_orginal()
        raw_img=func(org_i)
        return Image(img_i.name,raw_img)
    return inner_func

@dir_arg
def read_images(paths,nomalized=True):
    return [read_img(path_i) for path_i in paths]
    
@paths.path_args
def read_img(dir_path):
    raw_img=cv2.imread(str(dir_path),cv2.IMREAD_GRAYSCALE) 
    if(raw_img is None):
        return None#raise Exception("No image in path " + str(dir_path))
    img_i=Image(str(dir_path),raw_img)
    return img_i

def save_img(full_path,img):
    norm_const=250.0
    img=img.get_orginal()
    img*=norm_const
    img.astype(int)
    cv2.imwrite(str(full_path),img)

@ApplyToFiles(True)
@ApplyToFiles(False)
def rescale(in_path,out_path,new_dim=(60,60)):    
    img=cv2.imread(str(in_path))
    new_img=cv2.resize(img,new_dim)
    cv2.imwrite(str(out_path),new_img)

def make_imgs(in_path,norm=True,transform=None):
    img_dirs=paths.dirs.all_files(in_path)
    imgset=[read_img(path_i)
          for path_i in img_dirs]
    imgset=[img_i for img_i in imgset
                    if not (img_i is None)]
    if(norm):
        imgset=[ img_i/255.0
                 for img_i in imgset]
    print(type(imgset[0]))
    if(transform):
        imgset=transform(imgset)
    print(type(imgset[0]))
    return imgset

def unorm(imgset):
    return [ img_i*255.0
             for img_i in imgset]

def unify_img(x,scale=2):
    height=x.shape[1]
    width=x.shape[2]
    new_x=np.reshape(x,(scale*height,width))
    return new_x

def to_dataset(imgset,extract_cat,transform=None):

    if(extract_cat is None):
        cats=None#return np.array(imgset,dtype=float)
    else:
        cats=[ extract_cat(img_i.name) 
            for img_i in imgset]
    if(transform):
        print(type(transform))
        imgset=transform(imgset) 
    x=np.array(imgset,dtype=float)
    if(cats!=None):
        y=np.array(cats,dtype=float)
        return x,y
    else:
        return x

#def to_2D(imgset):
#    imgs2D=[ img_i.get_orginal()
#              for img_i in imgset]
#    conv=np.array(imgs2D)
#    conv=np.expand_dims(conv,1)
#    return conv

#def to_3D(imgset):
#    imgs3D=[ split_img(img_i.get_orginal())
#              for img_i in imgset]
#    vol=np.array(imgs3D)
#    return vol

#def to_proj(imgset):
#    imgs3D=[ split_img(img_i.get_orginal(),scale=3)
#              for img_i in imgset]
#    vol=np.array(imgs3D)
#    return vol

def split_img(x,scale=2):
    height=x.shape[0]
    width=x.shape[1]
    new_height=height/scale
    new_x=np.reshape(x,(scale,new_height,width))
    return new_x