import utils
import numpy as np

class Dataset(object):
    def __init__(self,instances):
        self.instances=instances

    def get_data(self):
        data=[inst.data for inst in self.instances] 
        return np.array(data)

    def get_reduced_data(self):
        data=[inst.reduced_data for inst in self.instances] 
        return np.array(data)

    def get_clusters(self):
        return [inst.cls for inst in self.instances]  

    def set_reduced(self,reduced_images):
        for i,redu_img in enumerate(reduced_images):
            self.instances[i].reduced_data=redu_img

    def set_cluster(self,clusters):
        for i,cls in clusters:
            self.instances[i].cls=cls

    def __len__(self):
        return len(self.instances)

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

def save_clusters(out_path,dataset):
    data=dataset.get_reduced_data()
    cls=dataset.get_clusters()
    utils.to_labeled_file(out_path,data,cls)

def save_reduce(out_path,dataset):
    data=dataset.get_reduced_data()
    utils.to_csv_file(out_path,data)
