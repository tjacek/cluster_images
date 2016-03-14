import utils.imgs,utils.files,utils.conf
import reduce_dim,clustering,basic
import deep.autoencoder as ae
import shape_context
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
    alg=INIT_FEATURES[config['init_features']]
    reduced_data=alg(in_path)
    print("init features")
    return reduced_data

def use_autoencoder(in_path):
    imgs=utils.imgs.read_img_dir(in_path)
    print("read data")
    reduced_img=ae.apply_autoencoder(imgs,config['ae_path'])
    return reduced_img

def use_shape_context(in_path):
    #imgs=utils.imgs.read_img_dir(in_path)
    img_paths=utils.files.get_files(in_path)
    red_imgs=[shape_context.get_shape_context(img_path_i)
                  for img_path_i in img_paths]
    red_imgs=[img_i for img_i in red_imgs
                      if img_i!=None]
    return red_imgs#,imgs

def use_basic(in_path):
    imgs=utils.imgs.read_img_dir(in_path)
    print("read data")
    basic_img=[basic.get_features(img_i) for img_i in imgs]
    return basic_img

INIT_FEATURES={'autoencoder':use_autoencoder,'shape_context':use_shape_context,'basic':use_basic}

def use_reduction(mf_data,config):
    alg_name=config.get('reduce_alg',None)
    if(alg_name==None):
        return mf_data
    reduce_alg=clustering.ALGS[alg_name]
    reduced_data=reduce_alg( mf_data,config)
    print("reduce data")
    return reduced_data

def use_clustering(mf_data,config):
    clust_alg=clustering.ALGS[config['cls_alg']]
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
    conf_path="conf/basic.cfg"
    cluster_images(conf_path)
