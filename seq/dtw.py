import utils.files as files
import numpy as np 
import seq

def wrap(in_path): 
    str_seqs=files.read_file(in_path)
    instances=seq.get_seqs(str_seqs)
    print(dwt_metric(instances[-1],instances[77]))
    print(instances)
    #k=action_to_vec_seq(str_seq,10)

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
    print(dwt)
    return dwt[n][m]

def d(v,d):
    return np.linalg.norm(v-d)
