import numpy as np

class SingleDTW(object):
    def __init__(self,conv,seqs,k=1):
        self.seqs=seqs
        self.k=k
        self.conv=conv

    def __call__(self,action):
        action_feat=self.conv(action)
        distance=[self.metric(x,seq_i) 
                    for seq_i in self.seqs]
        distance=np.array(distance)
        dist_inds=distance.argsort()[0:k]