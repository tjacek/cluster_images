#!/usr/bin/python
import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import utils
import utils.paths.files as files
import numpy as np
import split,to_dataset
import deep.lstm
from sklearn.metrics import classification_report,confusion_matrix
import utils.data
import deep.reader
import split

def make_model(train_dataset,make=True,n_epochs=0,p=0.5):
    if(make):
        hyper_params=deep.lstm.get_hyper_params(train_dataset)
        hyper_params['p']=p
        model=deep.lstm.compile_lstm(hyper_params)
    else:
        nn_reader=deep.reader.NNReader()
        model= nn_reader(nn_path,p)
    if(n_epochs!=0):
        return train_model(model,train_dataset,epochs=n_epochs)
    return model

def check_model(model,test_dataset):
    x=test_dataset['x']
    y_true=test_dataset['y']
    mask=test_dataset['mask']
    y_pred=[model.get_category(x_i,mask[i])
              for i,x_i in enumerate(x)]
    print(utils.data.find_errors(y_pred,test_dataset))
    check_prediction(y_pred,y_true)

class CorrectCond(object):
    def __init__(self, correct,seek_value=True):
        self.correct=correct
        self.seek_value=seek_value

    def __call__(self,dist,i):
        if(self.seek_value):
            return self.correct[i]
        else:
            return (not self.correct[i])

def check_distribution(model,test_dataset):
    x=test_dataset['x']
    y_true=test_dataset['y']
    mask=test_dataset['mask']
    y_pred=[model.get_category(x_i,mask[i])
              for i,x_i in enumerate(x)]
    dists=[model.get_distribution(x_i,mask[i])
              for i,x_i in enumerate(x)]
    correct=[ y_true_i==y_pred_i
              for y_true_i,y_pred_i in zip(y_true,y_pred)]

    dist_correct=[L2(dist_i)
                    for i,dist_i in enumerate(dists)
                      if(correct[i])]
    print(np.average(dist_correct))          
    print(np.average(dist_incorrect))          

def filter_dist(dist,cond):
    return [dist_i
             for i,dist_i in enumerate(dists)
                if(cond(dist,i))]

def L2(dists_i):
    return np.linalg.norm(dists_i,ord=2)

def train_model(model,dataset,epochs=10000):
    print(type(dataset))
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

def get_paths(dir_path,nn,seq='seq',lstm='lstm'):
    path=dir_path + nn+seq 
    nn_path=dir_path + nn+lstm 
    return path,nn_path

def create_dataset(path,nn_path):

    split_dataset= split.get_dataset(0,'modulo')

    dataset=to_dataset.seq_dataset(path,dataset_format='cp_dataset')
    print(dataset.params())
    new_dataset=to_dataset.masked_dataset(dataset)


    train,test=split_dataset(new_dataset)#split.person_dataset(new_dataset)
    model=make_model(train,True,n_epochs=100,p=0.0)
    model.get_model().save(nn_path)    
    check_model(model,test)

if __name__ == "__main__":
    #path='../cross/1_set/seq' #'../dataset2a/exp2/seq'
    #nn_path='../cross/1_set/lstm_full' #'../dataset2a/exp2/lstm_full'
    path='../ensemble3/12_nn/seq'
    nn_path='../ensemble3/12_nn/lstm'

    path,nn_path=get_paths('../ensemble3/','select/',seq='seq',lstm='lstm')
    create_dataset(path,nn_path)
#    split_dataset= split.get_dataset(0,'modulo')

#    dataset=to_dataset.seq_dataset(path,dataset_format='cp_dataset')
#    print(dataset.params())
#    new_dataset=to_dataset.masked_dataset(dataset)


#   train,test=split_dataset(new_dataset)#split.person_dataset(new_dataset)
#    model=make_model(train,True,n_epochs=20,p=0.0)
#    model.get_model().save(nn_path)    
#    check_model(model,test)
