from sklearn import cluster
from sklearn.cluster import AgglomerativeClustering as agglo
from sklearn.neighbors import kneighbors_graph

class DbscanAlg(object):
    def __init__(self, eps=0.1,min_samples=8):
        self.eps = eps
        self.min_samples=min_samples
        
    def __call__(self,data):
        db = cluster.DBSCAN(self.eps, self.min_samples).fit(data)
        return db.labels_

def KMeansAlg(object):
    def __init__(self, n_clusters=4):
        self.n_clusters=n_clusters

    def __call__(self,data):
        #cls=cluster.KMeans(n_clusters=clusters)
        cls=cluster.MiniBatchKMeans(n_clusters=clusters)
        res=cls.fit(data)
        return cls.labels_

def AgglomerAlg(data,config):
    def __init__(self,n_clusters=6,n_neighbors=10):
        self.n_clusters=n_clusters
        self.n_neighbors=n_neighbors
    
    def __call__(self,data):
        connectivity = kneighbors_graph(data, n_neighbors=self.n_neighbors, include_self=False)	
        cls=agglo(n_clusters=self.clusters, connectivity=connectivity,
                            linkage='ward')
        cls.fit(data)
        return cls.labels_