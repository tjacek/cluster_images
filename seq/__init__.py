#!/usr/bin/python
import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import utils
import utils.files as files
import numpy as np
import split,to_dataset
import deep.lstm
from sklearn.metrics import classification_report,confusion_matrix
import utils.data
import deep.reader

def make_model(train_dataset,make=True,n_epochs=0):
    if(make):
        hyper_params=deep.lstm.get_hyper_params(train_dataset)
        model=deep.lstm.compile_lstm(hyper_params)
    else:
        nn_reader=deep.reader.NNReader()
        model= nn_reader(nn_path,0.5)
    if(n_epochs!=0):
        return train_model(model,train_dataset,epochs=n_epochs)
    return model

def check_model(model,test_dataset):
    x=test_dataset['x']
    y_true=test_dataset['y']
    mask=test_dataset['mask']
    y_pred=[model.get_category(x_i,mask[i])
              for i,x_i in enumerate(x)]
    print(utils.data.find_errors(y_pred,test))
    check_prediction(y_pred,y_true)

def train_model(model,dataset,epochs=10000):
    x=get_batches(dataset['x'])
    y=get_batches(dataset['y'])
    print(dataset.keys())
    mask=get_batches(dataset['mask'])
    for j in range(epochs):
        cost=[]
        for i,x_i in enumerate(x):
            y_i=y[i]
            mask_i=mask[i]
            loss_i=model.train(x_i,y_i,mask_i)
            cost.append(loss_i)
        sum_j=sum(cost)/float(len(cost))
        print(str(j) + ' ' + str(sum_j))
    return model

def check_prediction(y_pred,y_true):
    print(classification_report(y_true, y_pred))
    print(confusion_matrix(y_true,y_pred))

def get_batches(x,batch_size=6):
    n_batches=len(x)/batch_size
    if((len(x) % batch_size)!=0):
        n_batches+=1
    return [x[i*batch_size:(i+1)*batch_size] 
               for i in range(n_batches)]

if __name__ == "__main__":
    path='../dataset1/exp2/seq/'
    nn_path='../dataset1/exp2/lstm_full'
    dataset=to_dataset.seq_dataset(path)
    new_dataset=to_dataset.masked_dataset(dataset)
    test,train=split.person_dataset(new_dataset)
    model=make_model(train,False,n_epochs=450)
    model.get_model().save(nn_path)    
    check_model(model,test)
