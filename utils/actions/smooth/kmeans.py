import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np 
import utils.actions.smooth
from sklearn import cluster

class ClusteringDisk(utils.actions.smooth.TimeSeriesTransform):
    def __init__(self, n_clusters=10):
        self.n_clusters=n_clusters

    def get_series_transform(self,frames):
        frames=np.array(frames)
        frames=frames.T    
        features=frames.tolist()
        means=[ self.get_means(feature_i) for feature_i in features]
        return NearestPointsDiskr(means)

    def get_means(self,feature_i):
        feature_i=np.array(feature_i)
        feature_i=np.expand_dims(feature_i,1)
        print(feature_i.shape)
        clust=cluster.MiniBatchKMeans(n_clusters=self.n_clusters,batch_size=300)
        clust.fit(feature_i)
        centers=clust.cluster_centers_
        centers=np.sort(np.array(centers),axis=0)
        return centers

class NearestPointsDiskr(object):
    def __init__(self,var):
        self.all_points=var

    def __call__(self,frame_i):
    	cords=frame_i.tolist()
        return [ self.get_cluster(cord_j,j) for j,cord_j in enumerate(cords)]

    def get_cluster(self,x_i,i):
        points=self.all_points[i]
        distance=[ np.abs(x_i-point_j) for point_j in points]
        return float(np.argmax(distance))

if __name__ == "__main__":
    in_path="../../AA_disk2/nn_0/"#nn_0"
    out_path="../../AA_disk2/test/"    
    clust_disk=ClusteringDisk()
    clust_disk(in_path,out_path)