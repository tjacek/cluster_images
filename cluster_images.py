import utils.imgs,utils.files,utils.conf
import reduce_dim,clustering
import deep.autoencoder as ae
import numpy as np
import cv2

def cluster_images(conf_path):
    config=utils.conf.read_config(conf_path)
    in_path=config["in_path"]
    out_path=config["out_path"]
    dataset,n_clusters=create_images(in_path,config)
    split_clusters(out_path,dataset,n_clusters)

def create_images(in_path,config):
    imgs=utils.imgs.read_img_dir(in_path)
    print(len(imgs))
    print("read data")
    reduced_img=ae.apply_autoencoder(imgs,config['ae_path'])#"../dataset/ae")
    print(reduced_img[0].shape)
    print("preproc")
    mf_data=reduce_dim.spectral_reduction( reduced_img,config)
    print("reduce data") 
    img_cls=clustering.dbscan(mf_data,config)#.kmeans(mf_data,config)
    print("cluster data")
    n_clusters=max(img_cls)
    dataset=[(img_i,cls_i) for img_i,cls_i in zip(imgs,img_cls)]
    save_clustering("clust.lb",mf_data,img_cls)
    return dataset,n_clusters

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
            utils.imgs.save_img(full_path,img_i)

if __name__ == "__main__":
    conf_path="conf/dataset6.cfg"
    cluster_images(conf_path)
