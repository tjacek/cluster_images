import utils
import numpy as np

class Instance(object):
    def __init__(self,name,data):
        self.name=name
        self.data=data
        self.reduced_data=None
        self.cls=None
     
    def __str__(self):
        return str(self.name)

    def file_id(self):
        return self.name.split("/")[-1]

def get_data(instances):
    data=[inst.data for inst in instances] 
    return np.array(data)

def get_reduced_data(instances):
    data=[inst.reduced_data for inst in instances] 
    return np.array(data)

def get_clusters(images):
    return [inst.cls for inst in images]  

def set_reduced(instances,reduced_images):
    for i,redu_img in enumerate(reduced_images):
        instances[i].reduced_data=redu_img

def set_cluster(instances,clusters):
    for i,cls in clusters:
        instances[i].cls=cls

def save_clusters(out_path,instances):
    data=get_reduced_data(instances)
    cls=get_clusters(instances)
    utils.to_labeled_file(out_path,data,cls)

def save_reduce(out_path,instances):
    data=get_reduced_data(instances)
    utils.to_csv_file(out_path,data)
