#!/usr/bin/python
import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import utils
import utils.files as files
import numpy as np
import split,to_dataset
import deep.lstm
from sklearn.metrics import classification_report,confusion_matrix

def make_model(train_dataset):
    hyper_params=deep.lstm.get_hyper_params(train_dataset)
    model=deep.lstm.compile_lstm(hyper_params)
    #model=deep.lstm.read_lstm('../dataset0/test')
    return train_model(model,train_dataset,epochs=2500)

def check_model(model,test_dataset):
    x=test_dataset['x']
    y_true=test_dataset['y']
    mask=test_dataset['mask']
    y_pred=[model.get_category(x_i,mask[i])
              for i,x_i in enumerate(x)]
    check_prediction(y_pred,y_true)

def train_model(model,dataset,epochs=10000):
    x=get_batches(dataset['x'])
    y=get_batches(dataset['y'])
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
    path='../dataset1/seq/'
    dataset=to_dataset.seq_dataset(path)
    new_dataset=to_dataset.masked_dataset(dataset)    
    #print(new_dataset['y'])
    train,test=split.person_dataset(new_dataset)
    print(train['y'])
    model=make_model(train)
    model.get_model().save('../dataset1/lstm_conv_')
    check_model(model,test)