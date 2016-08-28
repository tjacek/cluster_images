import numpy as np
import utils.text
from utils.data import make_dataset

def simple_dataset(dataset):
    return split_dataset(dataset,select_simple)

def person_dataset(dataset):
    select_person=SelectPerson(dataset['persons'])
    return split_dataset(dataset,select_person)

def split_dataset(dataset,select):
    train={}
    test={}
    for key_i,value_i in dataset.items():
        if(type(value_i)!=dict):
            print(key_i)        
            train[key_i]=select(dataset[key_i],n=0)
            test[key_i]=select(dataset[key_i],n=1)
        else:
            train[key_i]=dataset[key_i].copy()
            test[key_i]=dataset[key_i].copy()
    train['params']['n_batch']=len(train['y'])
    test['params']['n_batch']=len(test['y'])
    return train,test

def select_simple(instances,n=0,k=2):
    return [inst_i for i,inst_i in enumerate(instances)
                if((i % k)==n)]  

class SelectPerson(object):
    def __init__(self, persons):
        self.persons = persons
        
    def __call__(self,instances,n=0):
        return [inst_i for i,inst_i in enumerate(instances)
                 if((self.persons[i] % 2)==n)]  