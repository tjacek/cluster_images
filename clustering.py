from sklearn import cluster

def kmeans(data,config):
    clusters=int(config.get('clusters',4))
    cls=cluster.KMeans(n_clusters=clusters)
    res=cls.fit(data)
    return enumerate(cls.labels_)

def dbscan(data,config):
    db = cluster.DBSCAN(eps=0.1, min_samples=8).fit(data)
    return enumerate(db.labels_)
