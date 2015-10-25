import utils,read_frames,reduce_dim,clustering
import instances as inst
import scipy.misc as image

def cluster_images(in_path,out_path,n_clusters=12):
    images=create_images(in_path,n_clusters)
    inst.save_reduce("raw.csv",images)
    inst.save_clusters("cls.lb",images)
    split_clusters(out_path,images,n_clusters)

def create_images(in_path,n_clusters):
    images=read_frames.read_images(in_path)
    data=inst.get_data(images)
    reduced_data=reduce_dim.spectral_reduction(data)
    inst.set_reduced(images,reduced_data)
    img_cls=clustering.dbscan(reduced_data,n_clusters)
    inst.set_cluster(images,img_cls)
    return images

def split_clusters(out_path,images,n_clusters):
    utils.make_dir(out_path)
    cls_dirs=[out_path +"cls"+str(i)+"/" for i in range(n_clusters)]
    for c_dir in cls_dirs:
        utils.make_dir(c_dir)
    for img in images:
        if(img.cls>-1):
            orginal_img=image.imread(img.name)
            txt_id=img.file_id()
            print(img.cls)
            img_cls=cls_dirs[img.cls]
            full_path=img_cls+txt_id
            image.imsave(full_path,orginal_img)

if __name__ == "__main__":
    in_path="test_images/"
    out_path="out/"
    cluster_images(in_path,out_path)
