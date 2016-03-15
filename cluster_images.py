import utils.imgs,utils.files,utils.conf
import select_imgs
import numpy as np
import cv2

def cluster_images(conf_path):
    config=utils.conf.read_config(conf_path)
    in_path=config["in_path"]
    out_path=config["out_path"]
    dataset,n_clusters=create_images(in_path,config)
    split_clusters(out_path,dataset,n_clusters)

def create_images(in_path,config):
    imgs_=utils.imgs.read_img_dir(in_path)
    init_imgs=use_init_features(in_path,config)
    reduced_imgs=use_reduction(init_imgs,config)
    img_cls=use_clustering(reduced_imgs,config)
    n_clusters=max(img_cls)
    dataset=[(img_i,cls_i) for img_i,cls_i in zip(imgs_,img_cls)]
    save_clustering("clust.lb",reduced_imgs,img_cls)
    return dataset,n_clusters

def use_init_features(in_path,config):
    alg=select_imgs.INIT_FEATURES[config['init_features']]
    reduced_data=alg(in_path,config)
    reduced_data=[img_i for img_i in reduced_data
                        if img_i!=None]
    print("init features")
    return reduced_data

def use_reduction(mf_data,config):
    alg_name=config.get('reduce_alg',None)
    if(alg_name==None):
        return mf_data
    reduce_alg=select_imgs.ALGS[alg_name]
    reduced_data=reduce_alg( mf_data,config)
    print("reduce data")
    return reduced_data

def use_clustering(mf_data,config):
    clust_alg=select_imgs.CLUSTER[config['cls_alg']]
    img_cls= clust_alg(mf_data,config) #clustering.dbscan(mf_data,config)
    print("cluster data")
    return img_cls

def save_clustering(out_path, imgs,img_cls):
    lines=[utils.files.vector_string(img_i) + ",#" + str(cls_i) 
                             for img_i,cls_i in zip(imgs,img_cls)]
    #lines=[img_i +"#" + str(cls_i) for img_i,cls_i in dataset]
    lines=utils.files.array_to_txt(lines,"\n")
    utils.files.save_string(out_path,lines)

def split_clusters(out_path,dataset,n_clusters):
    n_clusters+=1
    utils.files.make_dir(out_path)
    cls_dirs=[out_path +"/cls"+str(i)+"/" for i in range(n_clusters)]
    for c_dir in cls_dirs:
        utils.files.make_dir(c_dir)
    i=0
    for img_i,cls_i in dataset:
        #cls_i=cls_i[1]
        #print(cls_i)
        if(cls_i>-1):
            txt_id="/fr"+str(i)+".jpg" #inst.file_id()
            i+=1
            full_path=out_path
            full_path=out_path+"/cls"+str(cls_i) +txt_id
            #print(full_path)
            utils.imgs.save_img(full_path,img_i,None)

if __name__ == "__main__":
    conf_path="conf/ae.cfg"
    cluster_images(conf_path)