import numpy as np
import scipy.misc as image
from instances import Instance
from sklearn.preprocessing import normalize
import imp
utils =imp.load_source("utils","/home/user/cls/cluster_images/utils.py")

def read_images(path):
    action_files=utils.get_dirs(path)
    action_files=utils.append_path(path,action_files)
    images=[]
    for action_path in action_files:
        images+=read_action(action_path)
    return instances.Dataset(images)

def read_action(action_path):
    all_files=utils.get_files(action_path)
    all_files=utils.append_path(action_path+"/",all_files)
    instances=[]
    for full_path in all_files:
        img=image.imread(full_path)
        img=img.flatten()
        img=img.astype(float)
        img/=max(img)
        img=np.reshape(img,(1,img.shape[0]))
        instances.append(Instance(full_path,img))
    return instances

def save_clusters(out_path,dataset):
    data=dataset.get_reduced_data()
    cls=dataset.get_clusters()
    utils.to_labeled_file(out_path,data,cls)

def save_reduce(out_path,dataset):
    data=dataset.get_reduced_data()
    utils.to_csv_file(out_path,data)
