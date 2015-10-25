import utils
import numpy as np

class Instance(object):
    def __init__(self,name,data):
        self.name=name
        self.data=data
        self.reduced_data=None

    def __str__(self):
        return str(self.name)

def get_data(instances):
    data=[inst.data for inst in instances] 
    return np.array(data)

def get_reduced_data(instances):
    data=[inst.reduced_data for inst in instances] 
    return np.array(data)

def set_reduced(instances,reduced_images):
    for i,redu_img in enumerate(reduced_images):
        instances[i].reduced_data=redu_img

def save_reduce(out_path,instances):
    data=get_reduced_data(instances)
    utils.to_csv_file(out_path,data)
