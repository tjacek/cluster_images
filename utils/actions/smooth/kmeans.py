import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np 
import utils.actions.smooth
from sklearn import cluster

class ClusteringDisk(utils.actions.smooth.TimeSeriesTransform):
    def __init__(self, n_clusters=10,dataset_format='cp_dataset'):
        super(ClusteringDisk, self).__init__(dataset_format)
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
        var_i=np.std(feature_i,axis=0)
#        print(var_i)
        if(var_i==0.0):
            print("Zero var")
            return None	
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
        if(points is None):
            return 0.0	
        distance=[ np.abs(x_i-point_j) for point_j in points]
        return float(np.argmin(distance))

if __name__ == "__main__":
    in_path="../../AA_disk3/norm_seqs/"
    out_path="../../AA_disk3/clust_seqs/"    
    clust_disk=ClusteringDisk()
    path_dec=utils.paths.dirs.ApplyToFiles(True)
    clust_disk= path_dec(clust_disk)
    clust_disk(in_path,out_path)