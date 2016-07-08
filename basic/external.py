import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np
import utils.timer
import utils.dirs as dirs
import utils.imgs as imgs
import utils.files as files
import deep_extr #as ae
from sklearn import manifold

def get_autoencoder_extractor(in_path):
    model=deep.read_model(in_path)
    ae_model=ae.build_autoencoder(model.hyperparams)
    ae_model.set_model(model)
    def extractor(data):
        return [ae_model.prediction(data_i)
                  for data_i in data]
    return extractor

@utils.timer.clock
def make_external(in_path,out_path):
    data=make_imgs(in_path)#[0:100]
    ae_extr=deep_extr.get_autoencoder_extractor("../dataset0a/ae")
    feat_dict=reduced_imgs(data,ae_extr)
    text_dict=files.dict_to_txt(feat_dict)
    files.save_string(out_path,text_dict)

def make_imgs(in_path,norm=False):
    img_dirs=dirs.all_files(in_path)
    imgset=[imgs.read_img(path_i)
          for path_i in img_dirs]
    if(norm):
        imgset=[ img_i/255.0
                 for img_i in imgset]
    return imgset

def transform_spectral(data,dim=30):
    X=np.array(data)
    embedder = manifold.SpectralEmbedding(n_components=dim, random_state=0,
        eigen_solver="arpack",n_neighbors=20)#neighbors)
    X_prim=embedder.fit_transform(X)
    return X_prim

#def get_autoencoder_transform(in_path):


def reduced_imgs(data,transform):
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
    #print(dir(deep))
    path_dir="../dataset0a/cats"
    out_dir="../dataset0a/ae.txt"
    make_external(path_dir,out_dir)
    #read_external(out_dir)
    #data=make_imgs(path_dir)[0:100]
    #print(len(data))
    #print(reduced_imgs(data))
    #X_prim=transform_dim(data)
    #print(X_prim.shape)