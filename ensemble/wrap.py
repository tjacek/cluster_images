import numpy as np

class FeatSeq(object):
    def __init__(self,conv):
        self.conv=conv

    def __call__(self,action):
        return [self.conv(img_i)  
                for img_i in action.img_seq]

class DTWcls(object):
    def __init__(self,seqs,k,metric):
        self.seqs=seqs
        self.k=k
        self.metric=metric

    def __call__(self,x):
        distance=[self.metric(x,seq_i) 
                    for seq_i in self.seqs]
        distance=np.array(distance)
        dist_inds=distance.argsort()[0:k]
