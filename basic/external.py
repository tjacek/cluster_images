import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np
import utils
import utils.dirs as dirs
import utils.imgs as imgs
from sklearn import manifold

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
    seq_files=utils.files.get_files(in_path,True)
    feat_dir={}
    for seq_i in seq_files:
        lines=utils.files.read_file(seq_i)
        pairs=[parse_line(line_i) for line_i in lines]
        for pair_i in pairs:
            feat_dir[pair_i[0]]=pair_i[1]
    print(feat_dir.keys())
    return feat_dir

def parse_line(line):
    line=line.split("#")
    name=utils.files.get_name(line[0])
    vector=line[1]
    vector=vector.replace(",\n","")
    vector=vector.split(",")
    vec_size=len(vector)
    num_vector=np.zeros((vec_size,),dtype=float)
    for i,cord_i in enumerate(vector):
    	num_vector[i]=float(cord_i)
    return (name,num_vector)

if __name__ == "__main__": 
    path_dir="../dataset0a/cats"
    data=make_imgs(path_dir)[0:100]
    print(len(data))
    print(reduced_imgs(data))
    #X_prim=transform_dim(data)
    #print(X_prim.shape)