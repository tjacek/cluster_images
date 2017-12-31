import numpy as np
import ensemble

class NNEnsemble(object):
    def __init__(self, deep_nns):
        self.deep_nns = deep_nns
    
    def __call__(self,seqs):
    	return [ self.get_category(seq_i) 
    	            for seq_i in seqs]
    
    def get_category(self,seq_i):
        return np.argmax(self.get_distribution(seq_i))
    
    def get_distribution(self,seq_i):
        dists=[nn_j.get_distribution(seq_i)
                for nn_j in self.deep_nns]
        dists=np.array(dists)
        return np.sum(dists, axis=0)

def read(nn_dir):
    nn_ensemble= NNEnsemble( ensemble.read_ensemble(nn_dir))	