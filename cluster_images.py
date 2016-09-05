import utils.imgs,utils.files,utils.conf,utils.actions
import select_imgs
import select_imgs.clustering
import select_imgs.tools
import numpy as np
import cv2
import utils.dirs
#from basic.external import external_features
from preproc import select_extractor

def cluster_images(conf_path):
    in_path=conf_path['img_path']
    out_path=conf_path['cls_path']
    data=utils.imgs.make_imgs(in_path,norm=True)
    extractor=select_extractor(conf_dict)
    imgset=[ extractor(img_i) for img_i in data ]
    cls_alg=select_imgs.clustering.KMeansAlg(5)
    labels=cls_alg(imgset)
    clusters=select_imgs.split_cls(labels,data)
    select_imgs.save_cls(out_path,clusters)
    #print(type(labels))

def create_cat_images(in_path,config): 
    init_imgs=use_init_features(in_path,config)
    #reduced_imgs=#use_reduction(init_imgs,config)
    #img_cls= #use_clustering(reduced_imgs,config)
    n_clusters=max(img_cls)
    dataset=[(img_i,cls_i) for img_i,cls_i in zip(imgs_,img_cls)]
    #save_clustering("clust.lb",reduced_imgs,img_cls)
    return dataset,n_clusters

if __name__ == "__main__":
    conf_path="conf/dataset3.cfg"
    conf_dict=utils.conf.read_config(conf_path)
    cluster_images(conf_dict)