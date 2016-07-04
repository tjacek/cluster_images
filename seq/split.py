import numpy as np
import utils.text
from to_dataset import make_dataset

def simple_dataset(dataset):
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

def select(instances,n=0,k=2):
    return [inst_i for i,inst_i in enumerate(instances)
                if((i % k)==n)]  

def person_dataset(instances):
    train=[]
    test=[] 
    for inst_i in instances:
        raw_person=inst_i.name.split("_")[0]
        person=utils.text.extract_number(raw_person)
        print("$$"+str(person)) 
        if((person % 2) ==0):
            train.append(inst_i)
        else:
            test.append(inst_i) 
    return train,test