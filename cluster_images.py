import utils,instances,read_frames,reduce_dim,clustering
import instances as inst

def cluster_images(path):
    images=read_frames.read_images(path)
    data=inst.get_data(images)
    reduced_data=reduce_dim.spectral_reduction(data)
    inst.set_reduced(images,reduced_data)
    img_cls=clustering.clustering_kmeans(reduced_data,4)
    inst.set_cluster(images,img_cls)
    inst.save_reduce("raw.csv",images)
    inst.save_clusters("cls.lb",images)

if __name__ == "__main__":
    path="test_images/"
    cluster_images(path)
