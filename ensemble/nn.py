import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np
import ensemble
import utils.actions.read

class NNEnsemble(object):
    def __init__(self, nns):
        self.nns = nns
    
    def __call__(self,seqs):
    	return [ self.get_category(seq_i) 
    	            for seq_i in seqs]
    
    def get_category(self,seq_i):
        return np.argmax(self.get_distribution(seq_i))
    
    def get_distribution(self,seq_i):
        dists=[nn_j.get_distribution(seq_i)
                for nn_j in self.nns]
        dists=np.array(dists)
        return np.sum(dists, axis=0)

class RNNCls(object):
    def __init__(self,rnn,conv_net):
        self.rnn=rnn
        self.conv_net=conv_net
    
    def get_distribution(self,action):
        seq_i=self.get_seq(action)
        return self.rnn.get_distribution(seq_i)

    def get_seq(self,action):
        if(type(action)!=utils.actions.Action):
            raise Exception("Non action type " + str(type(action)))
        return [self.conv(img_i)  
                for img_i in action.img_seq]

def apply_nn(in_path,rnn,dataset_format='cp_dataset'):
    action_reader=utils.actions.read.ReadActions(dataset_format,img_seq=True)
    actions=action_reader(in_path)
    y=[rnn(action_i) for action_i in actions]
        
def read_rnn(rnn_path,conv_path):
    rnns=ensemble.read_ensemble(rnn_path,with_id=True)
    conv_nets=ensemble.read_ensemble(conv_path,with_id=True)
    clss=[RNNCls(rnns[key_i],conv_nets[key_i]) 
            for key_i in rnns.keys()]
    return NNEnsemble(clss)


if __name__ == "__main__":
    in_path="../../AArtyk/MSR/time/"
    rnn_path="../../AArtyk/all_models/"
    conv_path="../../AArtyk_exp/lstm/"
    rnn=read_rnn(rnn_path,conv_path)
    print("**********************************")
    apply_nn(in_path,rnn)