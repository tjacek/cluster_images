import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np 
from collections import Counter
from seq.to_dataset import seq_dataset
import seq,split
import utils.paths as paths
from utils.timer import clock 

@paths.path_args
def use_dtw(dataset_path):
    dataset=seq_dataset(path)
    train,test=split.person_dataset(dataset)
    y_pred=wrap(train,test)
    seq.check_prediction(y_pred,test['y'])

@clock
def wrap(train,test): 
    return [knn(test_i,train) 
              for test_i in test['x']]

def knn(new_x,train_dataset,k=3):
    distance=[dtw_metric(new_x,x_i) 
              for x_i in train_dataset['x']]
    distance=np.array(distance)
    dist_inds=distance.argsort()[0:k]
    y=   train_dataset['y']
    nearest=[y[i] for i in dist_inds]
    print(nearest)
    count =Counter(nearest)
    new_cat=count.most_common()[0][0]
    print(new_cat)
    return new_cat

def dtw_metric(s,t):
    n=len(s)
    m=len(t)
    dwt=np.zeros((n+1,m+1),dtype=float)
    for i in range(1,n+1):
        dwt[i][0]=np.inf
    for i in range(1,m+1):
        dwt[0][i]=np.inf
    dwt[0][0]=0.0
    for i in range(1,n+1):
        for j in range(1,m+1):
            cost=d(s[i-1],t[j-1])
            dwt[i,j]=cost+min([dwt[i-1][j],dwt[i][j-1],dwt[i-1][j-1]])
    return dwt[n][m]

def d(v,d):
    return np.linalg.norm(v-d)

if __name__ == "__main__":
    path='../dataset1/seq/'
    use_dtw(path)