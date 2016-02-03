import utils.imgs,utils.files
import reduce_dim,clustering
import deep.autoencoder#,reduce_dim#,#clustering,preproc
#import scipy.misc as image
import cv2

def cluster_images(in_path,out_path,n_clusters=4):
    dataset=create_images(in_path,n_clusters)
    #data.save_reduce("raw.csv",images)
    #data.save_clusters("cls.lb",images)
    split_clusters(out_path,dataset)

def create_images(in_path,n_clusters):
    imgs=utils.imgs.read_img_dir(in_path)
    print("read data")
    #preproc.deep_reduce(dataset)
    reduced_img=deep.autoencoder.apply_autoencoder(imgs,"../dataset/ae")
    print(reduced_img[0].shape)
    print("preproc")
   # images=dataset.get_data()
   # print(images.shape)
    mf_data=reduce_dim.spectral_reduction( reduced_img)
   # dataset.set_reduced(reduced_data)
    print("reduce data") 
    img_cls=clustering.dbscan(mf_data,n_clusters)
    print("cluster data")
    print(img_cls)
    dataset=[(img_i,cls_i) for img_i,cls_i in zip(imgs,img_cls)] # dataset.set_cluster(img_cls)
    return dataset

def split_clusters(out_path,dataset):
    utils.files.make_dir(out_path)
    n_clusters= 10#dataset.get_number_of_clusters()
    cls_dirs=[out_path +"/cls"+str(i)+"/" for i in range(n_clusters)]
    for c_dir in cls_dirs:
        utils.files.make_dir(c_dir)
    i=0
    for img_i,cls_i in dataset:#.instances:
        cls_i=cls_i[1]
        print(cls_i)
        print(type(img_i))
        if(cls_i>-1):
            #orginal_img=image.imread(inst.name)
            txt_id="/fr"+str(i)+".jpg" #inst.file_id()
            i+=1
            full_path=out_path
            #print(inst.cls)
            #img_cls=cls_dirs[inst.cls]
            full_path=out_path+"/cls"+str(cls_i) +txt_id
            print(full_path)
            img_i=img_i.reshape(90,40)
            cv2.imwrite(full_path,img_i)
            #image.imsave(full_path,orginal_img)

if __name__ == "__main__":
    in_path="../dataset/imgs"
    obj_path="../dataset/ae"
    out_path="../dataset/out"
    cluster_images(in_path,out_path)
