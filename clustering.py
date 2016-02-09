from sklearn import cluster

def kmeans(data,config):
    clusters=int(config.get('clusters',4))
    cls=cluster.KMeans(n_clusters=clusters)
    res=cls.fit(data)
    return cls.labels_

def dbscan(data,config):
    eps=float(config.get('eps',0.1))
    min_samples=int(config.get('min_samples',8))
    db = cluster.DBSCAN(eps, min_samples).fit(data)
    return db.labels_
