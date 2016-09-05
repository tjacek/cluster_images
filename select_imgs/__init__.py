import clustering

def split_cls(labels,data):
    n_cats=max(labels)+1
    clusters=[[] for i in range(n_cats)]
    for label_i,data_i in zip(labels,data):
        clusters[label_i].append(data_i)
    for cluster_i in clusters:
        print(len(cluster_i))
    return clusters