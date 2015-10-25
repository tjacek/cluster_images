from sklearn import cluster

def kmeans(data,clusters=4):
    cls=cluster.KMeans(n_clusters=clusters)
    res=cls.fit(data)
    return enumerate(cls.labels_)

def dbscan(data,clusters=4):
    db = cluster.DBSCAN(eps=0.3, min_samples=8).fit(data)
    return enumerate(db.labels_)
