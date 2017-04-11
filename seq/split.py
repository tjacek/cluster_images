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

def simple_dataset(dataset):
    return split_dataset(dataset,select_simple)

def person_dataset(dataset):
    select_person=SelectPerson(dataset['persons'])
    return split_dataset(dataset,select_person)

def split_dataset(dataset,select):
    train_odd={}
    test_even={}
    for key_i,value_i in dataset.items():
        if(type(value_i)!=dict):
            print(key_i)        
            test_even[key_i]=select(dataset[key_i],n=0)
            print(select.persons)
            train_odd[key_i]=select(dataset[key_i],n=1)
        else:
            test_even[key_i]=dataset[key_i].copy()
            train_odd[key_i]=dataset[key_i].copy()
    train_odd['params']['n_batch']=len(train_odd['y'])
    test_even['params']['n_batch']=len(test_even['y'])
    return train_odd,test_even

def select_simple(instances,n=0,k=2):
    return [inst_i for i,inst_i in enumerate(instances)
                if((i % k)==n)]  

class SelectPerson(object):
    def __init__(self, persons):
        self.persons = [ (person_i % 2) 
                         for person_i in persons]
        
    def __call__(self,instances,n=0):
        return [inst_i for i,inst_i in enumerate(instances)
                 if(self.persons[i]==n)]  