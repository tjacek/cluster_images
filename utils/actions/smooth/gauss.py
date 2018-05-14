import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np 
import utils.actions.smooth
import utils.actions.tools
import utils.actions.smooth.kmeans

from sklearn import cluster
from sklearn.mixture import GaussianMixture

class GaussDisk(utils.actions.smooth.TimeSeriesTransform):
    def __init__(self,max_k=20,dataset_format='cp_dataset'):
        super(GaussDisk, self).__init__(dataset_format)
        self.max_k=max_k

    def get_series_transform(self,frames):   
        frames=np.array(frames)#.T
        features=utils.actions.tools.to_features(frames)
        models=[self.select_model(feature_i) for feature_i in features]
        return utils.actions.smooth.kmeans.NearestPointsDiskr(models)
    def select_model(self,data):
        data=np.array(data)
        data=np.expand_dims(data,axis=1)

        if(np.var(data)==0.0):
            return None      	
        models=[self.fit_model(i+1,data) 
                    for i in range(self.max_k)]

        crit=[model_i.bic(data) for model_i in models]            
        print(crit)
        s_model=models[np.argmin(crit)]
        centers=np.sort(s_model.means_,axis=0)
        print("Selected number %d" % len(centers))
        print(centers)
        return centers
#        return models[s]

    def fit_model(self,i,data):
        model_i=GaussianMixture(n_components=i,
                covariance_type='spherical', max_iter=20, random_state=0)
        model_i.fit(data)
        return model_i

class GaussCluster(object):
    def __init__(self,mixtures):
        self.mixtures=mixtures

    def __call__(self,frame_i):
    	cords=frame_i.tolist()
        return [ self.get_value(i,cord_j) 
                    for i,cord_j in enumerate(cords)]

    def get_value(self,i,cord_j):
        mix_i=self.mixtures[i]
        if(mix_i is None):
            return 0.0  	
        return mix_i.predict(cord_j)[0] 	

if __name__ == "__main__":
    in_path="../../AA_disk5/norm_seqs/"
    out_path="../../AA_disk5/clust_seqs/"    
    clust_disk=GaussDisk()
    path_dec=utils.paths.dirs.ApplyToFiles(True)
    clust_disk= path_dec(clust_disk)
    clust_disk(in_path,out_path)