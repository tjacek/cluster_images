import utils.imgs,utils.files,utils.conf
import select_imgs
import select_imgs.clustering
import select_imgs.tools
import numpy as np
import cv2
import utils.dirs
from preproc import select_extractor

def cluster_images(conf_path):
    in_path=conf_path['img_path']
    out_path=conf_path['cls_path']
    data=utils.imgs.make_imgs(in_path,norm=True)
    extractor=select_extractor(conf_dict)
    imgset=[ extractor(img_i) for img_i in data ]
    cls_alg=select_imgs.clustering.DbscanAlg()
    #cls_alg=select_imgs.clustering.KMeansAlg(20)#(5)
    labels=cls_alg(imgset)
    unorm_data=utils.imgs.unorm(data)
    clusters=select_imgs.split_cls(labels,unorm_data)
    select_imgs.save_cls(out_path,clusters)
    #print(type(labels))

if __name__ == "__main__":
    conf_path="conf/dataset3.cfg"
    conf_dict=utils.conf.read_config(conf_path)
    cluster_images(conf_dict)