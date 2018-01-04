import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np 
import utils.actions,utils.actions.read
from sklearn.metrics import classification_report,confusion_matrix
import utils.data

def seq_dataset(in_path,masked=False,dataset_format='cp_dataset'):
    print(str(in_path))
    action_reader=utils.actions.read.ReadActions(dataset_format,img_seq=False)
    actions= action_reader(in_path)  
    test= utils.actions.raw_select(actions,0)
    train= utils.actions.raw_select(actions,1)    
    train=make_dataset(train,masked)
    test=make_dataset(test,masked)
    return train,test

def make_dataset(actions,masked=False):
    x=[np.array(action_i.img_seq)
        for action_i in actions]
    x=np.array(x)
    y=[action_i.cat for action_i in actions]
    names=[action_i.name for action_i in actions]
    persons=[action_i.person for action_i in actions]
    extract_cat=utils.data.ExtractCat(lambda x:x)
    y=[ extract_cat(y_i) for y_i in y]
    basic_dataset={'x':x ,'y':y,'names':names,'persons':persons}
    if(masked):
        return masked_dataset(basic_dataset)
    else:
        return basic_dataset

def masked_dataset(dataset):
    x=dataset['x']
    y=dataset['y']
    names=dataset['names']
    params= utils.data.make_params(x,y)
    mask=make_mask(x,params['n_batch'],params['max_seq'])
    x_masked=make_masked_seq(x,params['max_seq'],params['seq_dim'])
    new_dataset={'x':x_masked,'y':dataset['y'],'mask':mask,
                 'persons':dataset['persons'],'params':params,
                 'names':names}
    return new_dataset#SeqDataset(new_dataset)

def make_mask(x,n_batch,max_seq):
    mask = np.zeros((n_batch, max_seq),dtype=float)
    for i,seq_i in enumerate(x):
        seq_i_len=utils.data.seq_len(seq_i)
        mask[i][:seq_i_len]=1.0
    return mask

def make_masked_seq(x,max_seq,seq_dim):
    
    def masked_seq(seq_i):
        if(seq_i.shape[0]>max_seq):
            return np.ones( (seq_i.shape[0],seq_dim))
        seq_i_len=utils.data.seq_len(seq_i)
        new_seq_i=np.zeros((max_seq,seq_dim))
        new_seq_i[:seq_i_len]=seq_i[:seq_i_len]
        return new_seq_i
    return [masked_seq(seq_i) for seq_i in x]    

def check_prediction(y_pred,y_true):
    print(classification_report(y_true, y_pred,digits=4))
    print(confusion_matrix(y_true,y_pred))

def unify_dataset(dataset1,dataset2):
    new_dataset={}
    for key_i,value_i in dataset1.items():        
        if(type(value_i)==list):
            new_value=value_i+dataset2[key_i]
        elif(type(value_i)==np.ndarray):
            new_value=np.concatenate((value_i,dataset2[key_i]),axis=0)
        else:
            new_value=value_i
        new_dataset[key_i]=new_value 
    return new_dataset