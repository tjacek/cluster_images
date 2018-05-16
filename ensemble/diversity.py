import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np
import ensemble,ensemble.nn,ensemble.rnn
import utils.actions,utils.actions.read
from sets import Set

def compute_diversity(in_path,lstm_ens,dataset_format='cp_dataset',img_seq=True):
    action_reader=utils.actions.read.ReadActions(dataset_format,img_seq=img_seq)
    actions=action_reader(in_path)
    test= utils.actions.raw_select(actions,0)    
#    names=[name_i for name_i in names]
    labels=[ensemble.nn.get_labels(rnn_i,test)  
                for rnn_i in lstm_ens.nns]
    cls_mistakes=[ get_mistakes(labels_i) for labels_i in labels]
    diversity=[np.argsort([len(mistake_i.intersection(mistake_j))
                        for mistake_i in cls_mistakes])
                    for mistake_j in cls_mistakes]
    print(diversity)
    print(np.argsort(diversity))

def get_mistakes(labels_i):
    print("labels")
    pred_y,true_y=labels_i[0],labels_i[1]
    mistakes=[]
    for i,true_i in enumerate(true_y):
        if(true_i!=pred_y[i]):
            mistakes.append(i)
    return Set(mistakes)

if __name__ == "__main__":
    in_path="../../Documents/AA/united/"
    lstm_path="../../Documents/AA/smooth_lstm/"
    rnn_ensemble=ensemble.rnn.make_rnn_ensemble(in_path,lstm_path)
    in_path="../../Documents/AA/united/nn_0"
    compute_diversity(in_path,rnn_ensemble,img_seq=False)