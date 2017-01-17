import numpy as np
from seq.dtw import dtw_metric

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

    def dataset_scale(self):
        pair_dist=[dtw_metric(seq_i,seq_j)
            for seq_i in self.seqs
              for seq_j in self.seqs]