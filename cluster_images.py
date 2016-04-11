import utils.imgs,utils.files,utils.conf,utils.actions
import select_imgs
import select_imgs.tools
import numpy as np
import cv2
from utils.dirs import dir_arg, ApplyToFiles

def cluster_images(conf_path):
    config=utils.conf.read_config(conf_path)
    in_path=config["in_path"]
    out_path=config["out_path"]
    @ApplyToFiles(True)
    def inner_func(in_cat,out_cat):
        print(str(in_cat)+"\n"+str(out_path)+"\n")
        new_config=config.copy()
        new_config["in_path"]=str(in_cat)
        new_config["out_path"]=str(out_cat)
        if(config['init_features']=='autoencoder'):
            new_config["ae_path"]=config["ae_path"]+"/"+in_cat.get_name()
        imgs=utils.imgs.read_images(in_cat)
        print("Number of images %i",len(imgs))
        dataset,n_clusters=select_imgs.create_images(in_cat,new_config,imgs)
        select_imgs.tools.split_clusters(new_config,dataset,n_clusters)
    inner_func(in_path,out_path)

def create_cat_images(in_path,config): 
    init_imgs=use_init_features(in_path,config)
    #reduced_imgs=#use_reduction(init_imgs,config)
    #img_cls= #use_clustering(reduced_imgs,config)
    n_clusters=max(img_cls)
    dataset=[(img_i,cls_i) for img_i,cls_i in zip(imgs_,img_cls)]
    #save_clustering("clust.lb",reduced_imgs,img_cls)
    return dataset,n_clusters

def category_cluster(action_path):
    actions,cats=utils.actions.get_action_dataset(action_path)
    cat_i=utils.data.extract_cat(actions,cats,1)
    print(len(cat_i))

if __name__ == "__main__":
    conf_path="conf/dataset9.cfg"
    #cluster_images(conf_path)
    cluster_images(conf_path)