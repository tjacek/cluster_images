import numpy as np
import utils.text
from utils.data import make_dataset
import seq.to_dataset

class SplitDataset(object):
    def __init__(self,key,cond):
        self.key=key
        self.cond=cond 
    
    def __call__(self,dataset):
        inst=dataset.to_instances()
        test_inst=[inst_i 
                      for inst_i in inst
                        if(self.cond(inst_i[self.key]))]    
        train_inst=[inst_i 
                      for inst_i in inst
                        if(not self.cond(inst_i[self.key]))]

        test_dataset=seq.to_dataset.from_instances(test_inst,dataset['params'])
        train_dataset=seq.to_dataset.from_instances(train_inst,dataset['params'])
        print(len(test_dataset))
        print(len(train_dataset))
        return train_dataset,test_dataset

def get_modulo_dataset():
    selector=lambda person_i: (person_i %2)==0
    return SplitDataset('persons', selector)

def get_equal_dataset(k=1):
    selector=lambda person_i: person_i==k
    #def selector(person_i):
        #print(person_i)
        #print(person_i==k)
    #    return person_i==k
    return SplitDataset('persons', selector)