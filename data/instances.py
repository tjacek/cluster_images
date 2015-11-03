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

    def get_name(self,i):
        return self.instances[i].file_id()

    def get_number_of_clusters(self):
        cls=self.get_clusters()
        return max(cls)+1
 
    def __len__(self):
        return len(self.instances)

    def __str__(self):
        return "Dataset: " + str(len(self))

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
