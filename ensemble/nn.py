import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
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

class RNNCls(object):
    def __init__(self,rnn,conv_net):
        self.rnn=rnn
        self.conv_net=conv_net
    
    def __call__(self,action):
        seq_i=self.get_seq(action)
        self.rnn.get_distribution(self,x,mask)

    def get_seq(self,action):
        if(type(action)!=utils.actions.Action):
            raise Exception("Non action type " + str(type(action)))
        return [self.conv(img_i)  
                for img_i in action.img_seq]

def read_rnn(rnn_path,conv_path):
    rnns=ensemble.read_ensemble(rnn_path,with_id=True)
    conv_nets=ensemble.read_ensemble(conv_path,with_id=True)

if __name__ == "__main__":
    rnn_path="../../AArtyk/all_models/"
    conv_path="../../AArtyk_exp/lstm/"
    read_rnn(rnn_path,conv_path)