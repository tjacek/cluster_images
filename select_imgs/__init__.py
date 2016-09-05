import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import clustering
import utils.dirs
import utils.paths

def split_cls(labels,data):
    n_cats=max(labels)+1
    clusters=[[] for i in range(n_cats)]
    for label_i,data_i in zip(labels,data):
        clusters[label_i].append(data_i)
    #for cluster_i in clusters:
    #    print(len(cluster_i))
    return clusters

def save_cls(out_path,clusters):
    utils.dirs.make_dir(out_path)
    #cls_paths=[out_path+"/"+"cls"+str(i)
    #            for i,cls_i in enumerate(clusters)]
    for i,cls_i in enumerate(clusters):
        cls_path_i=str(out_path)+"/"+"cls"+str(i)
        cls_path_i= utils.paths.Path(cls_path_i)
        utils.dirs.make_dir(cls_path_i)
        for img_i in clusters[i]:
            img_i.save(cls_path_i)

