import utils.text

def simple_dataset(instances):
    train=[]
    test=[]	
    for i,inst_i in enumerate(instances):
        if((i % 2) ==0):
            train.append(inst_i)
        else:
            test.append(inst_i)	
    return train,test

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