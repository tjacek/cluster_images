from sklearn import cluster
from sklearn.cluster import AgglomerativeClustering as agglo
from sklearn.neighbors import kneighbors_graph

ALGS={'kmeans':kmeans,'dbscan':dbscan,'agglomer':agglomer}

def kmeans(data,config):
    clusters=int(config.get('clusters',4))
    #cls=cluster.KMeans(n_clusters=clusters)
    cls=cluster.MiniBatchKMeans(n_clusters=clusters)
    res=cls.fit(data)
    return cls.labels_

def dbscan(data,config):
    eps=float(config.get('eps',0.1))
    min_samples=int(config.get('min_samples',8))
    db = cluster.DBSCAN(eps, min_samples).fit(data)
    return db.labels_

def agglomer(data,config):
    clusters=int(config.get('clusters',6))
    connectivity = kneighbors_graph(data, n_neighbors=10, include_self=False)	
    cls=agglo(n_clusters=clusters, connectivity=connectivity,
                               linkage='ward')
    cls.fit(data)
    return cls.labels_