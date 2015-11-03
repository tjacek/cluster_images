import utils,data,reduce_dim,clustering,preproc
import scipy.misc as image

def cluster_images(in_path,out_path,n_clusters=4):
    images=create_images(in_path,n_clusters)
    data.save_reduce("raw.csv",images)
    data.save_clusters("cls.lb",images)
    split_clusters(out_path,images)

def create_images(in_path,n_clusters):
    dataset=data.read_images(in_path)
    print("read data")
    preproc.deep_reduce(dataset)
    print("preproc")
    images=dataset.get_data()
    print(images.shape)
    reduced_data=reduce_dim.spectral_reduction(images)
    dataset.set_reduced(reduced_data)
    print("reduce data") 
    img_cls=clustering.dbscan(reduced_data,n_clusters)
    print("cluster data")
    dataset.set_cluster(img_cls)
    return dataset

def split_clusters(out_path,dataset):
    utils.make_dir(out_path)
    n_clusters=dataset.get_number_of_clusters()
    cls_dirs=[out_path +"cls"+str(i)+"/" for i in range(n_clusters)]
    for c_dir in cls_dirs:
        utils.make_dir(c_dir)
    for inst in dataset.instances:
        print(inst.cls)
        if(inst.cls>-1):
            orginal_img=image.imread(inst.name)
            txt_id=inst.file_id()
            print(inst.cls)
            img_cls=cls_dirs[inst.cls]
            full_path=img_cls+txt_id
            image.imsave(full_path,orginal_img)

if __name__ == "__main__":
    path="/home/user/cls/"
    in_path=path+"test/"
    out_path=path+"out2/"
    cluster_images(in_path,out_path)
