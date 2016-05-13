import utils.files as files
from utils.timer import clock 
import split
import numpy as np 
import seq
from collections import Counter
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import feats
from sklearn.manifold import MDS

def wrap_seq(in_path): 
    str_seqs=files.read_file(in_path)
    instances=seq.get_seqs(str_seqs)
    wrap(instances)

@clock
def wrap(instances): 
    test,train=split.person_dataset(instances)
    correct=[test_i.cat for test_i in test]
    pred=[knn(test_i,train) for test_i in test]
    cat_to_int=feats.int_cats(correct)
    print(classification_report(correct, pred))
    correct=[cat_to_int[cat_i] for cat_i in correct]
    pred=[cat_to_int[cat_i] for cat_i in pred]
    print(confusion_matrix(correct,pred))

def knn(inst,instances,k=1):
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

def visualize_dwt(instances):
    n_samples=len(instances)
    similarities=np.zeros((n_samples,n_samples))
    for i,inst_i in enumerate(instances):
        print(i)
        for j,inst_j in enumerate(instances):
            similarities[i][j]=dwt_metric(inst_i,inst_j)        
    seed = np.random.RandomState(seed=3)
    nmds = MDS(n_components=2, metric=False, max_iter=3000, eps=1e-12,
                    dissimilarity="precomputed", random_state=seed, n_jobs=1,
                    n_init=1)
    pos = nmds.fit(similarities).embedding_
    X=np.array(pos)
    y=[ seq_i.cat for seq_i in instances]
    print(type(instances[0]))
    print(type(pos))
    return X,y
    #npos = nmds.fit_transform(similarities, init=pos)