import utils.text
from to_dataset import make_dataset

def simple_dataset(dataset):
    x_train=select(dataset['x'],n=0)
    x_test=select(dataset['x'],n=1)
    y_train=select(dataset['y'],n=0)
    y_test=select(dataset['y'],n=1) 
    return make_dataset(x_train,y_train),make_dataset(x_test,y_test)

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