import utils.files as files
import numpy as np 
import seq
from collections import Counter
from sklearn.metrics import classification_report

def wrap(in_path): 
    str_seqs=files.read_file(in_path)
    instances=seq.get_seqs(str_seqs)
    test,train=select_dataset(instances)
    correct=[test_i.cat for test_i in test]
    pred=[knn(test_i,train) for test_i in test]
    print(classification_report(correct, pred))


def knn(inst,instances,k=6):
    dists=[dwt_metric(inst,inst_i) for inst_i in instances]
    dists=np.array(dists)
    dist_inds=dists.argsort()[0:k]
	#nearest=[dists[i] for i in dist_inds]
    nearest=[instances[i].cat for i in dist_inds]
    print(nearest)
    count =Counter(nearest)
    new_cat=count.most_common()[0][0]
    return new_cat

def dwt_metric(s,t):
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


def select_dataset(instances):
    train=[]
    test=[]	
    for i,inst_i in enumerate(instances):
        if((i % 2) ==0):
            train.append(inst_i)
        else:
            test.append(inst_i)	
    return train,test
