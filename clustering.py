from sklearn import cluster

def clustering_kmeans(data,clusters=4):
    cls=cluster.KMeans(n_clusters=clusters)
    res=cls.fit(data)
    print(len(data))
    print(len(cls.labels_))
    return enumerate(cls.labels_)
