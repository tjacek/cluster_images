import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import utils
import utils.files as files
import numpy as np
import to_dataset
import deep.lstm

def train(model,dataset,epochs=10):
    x=get_batches(dataset['x'])
    y=get_batches(dataset['y'])
    mask=get_batches(dataset['mask'])
    for j in range(epochs):
        for i,x_i in enumerate(x):
            y_i=y[i]
            mask_i=mask[i]
            #print(len(x))
            print(model.train(x_i,y_i,mask_i))

def get_batches(x,batch_size=5):
    n_batches=len(x)/batch_size
    if((len(x) % batch_size)!=0):
        n_batches+=1
    return [x[i*batch_size:(i+1)*batch_size] 
               for i in range(n_batches)]

if __name__ == "__main__":
    path='../dataset0/seq/'
    dataset=to_dataset.seq_dataset(path)
    new_dataset=to_dataset.masked_dataset(dataset)
    hyper_params=deep.lstm.get_hyper_params(new_dataset)
    lstm_equ,input_vars=deep.lstm.make_LSTM(hyper_params)
    model=deep.lstm.compile_lstm(lstm_equ,input_vars,hyper_params)
    train(model,new_dataset)