import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np
import utils.timer
import utils.dirs as dirs
import utils.imgs as imgs
import utils.files as files
from sklearn import manifold

@utils.timer.clock
def make_external(in_path,out_path):
    data=make_imgs(in_path)#[0:100]
    feat_dict=reduced_imgs(data)
    text_dict=files.dict_to_txt(feat_dict)
    files.save_string(out_path,text_dict)

def make_imgs(in_path):
    img_dirs=dirs.all_files(in_path)
    imgset=[imgs.read_img(path_i)
          for path_i in img_dirs]
    return imgset

def transform_dim(data,dim=30):
    X=np.array(data)
    embedder = manifold.SpectralEmbedding(n_components=dim, random_state=0,
        eigen_solver="arpack",n_neighbors=20)#neighbors)
    X_prim=embedder.fit_transform(X)
    return X_prim

def reduced_imgs(data,transform=transform_dim):
    names=dict([ (data_i.name,i) 
                  for i,data_i in enumerate(data)])
    data_prim=transform(data)
    feat_dict=dict([(name_i,data_prim[i])
                      for name_i,i in names.items()])
    return feat_dict

def read_external(in_path):
    text='\n'.join(files.read_file(in_path))
    feat_dict=files.txt_to_dict(text)
    def get_features(img_i):
        return feat_dict[img_i.name]
    return get_features

if __name__ == "__main__": 
    path_dir="../dataset0a/cats"
    out_dir="../dataset0a/spectral2.txt"
    #make_external(path_dir,out_dir)
    read_external(out_dir)
    #data=make_imgs(path_dir)[0:100]
    #print(len(data))
    #print(reduced_imgs(data))
    #X_prim=transform_dim(data)
    #print(X_prim.shape)