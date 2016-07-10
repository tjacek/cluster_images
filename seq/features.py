import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np
import utils.paths as paths
import utils.dirs as dirs
import utils.imgs as imgs
import utils.files as files
import utils.pcloud as pcloud
import basic
import basic.external as ext

@paths.path_args
def extract_features(in_path):
    out_path=in_path.copy().set_name('seq')
    print(str(out_path)) 
    transform_seq(in_path,out_path)

@dirs.apply_to_dirs
def transform_seq(in_path,out_path):
    extractor=basic.get_features
    imgs_seq=imgs.read_images(in_path)
    seq=[extractor(img_i) 
                for img_i in imgs_seq]
    txt=files.seq_to_string(seq)
    print(str(in_path))
    print(str(out_path))
    files.save_string(str(out_path)+'.txt',txt)

#def extr_feat(img_i):
    #feat=ext.read_external("../dataset0a/spectral2.txt")
    #cloud_i=pcloud.make_point_cloud(img_i)
    #cloud_i=pcloud.normalized_cloud(cloud_i)
    #com=cloud_i.center_of_mass()
    #sk=basic.skewness_features(img_i,cloud_i)
    #return feat(img_i)#read_externalnp.concatenate([com,sk])

if __name__ == "__main__":
    path='../dataset0/cats/'
    extract_features(path)