import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np
import deep.reader,deep.tools as tools
import utils.actions.read

class BinaryEnsemble(object):
    def __init__(self,binary_networks):
        self.binary_networks=binary_networks

    def get_seq(self,actions):
        return [action_i.transform(self,img_seq=False) 
                    for action_i in actions]    	
    
    def __call__(self,img_i):
        def dist_helper(nn_j):
            prob_j=nn_j.get_distribution(img_i)[1] 
            if(prob_j<0.05):
                return 0.0
            return prob_j
        feats=[dist_helper(nn_j)
                for nn_j in self.binary_networks]
        return np.array(feats)

def ensemble_seq(in_path,nn_dir,out_path,dataset_format='cp_dataset'):
    action_reader=utils.actions.read.ReadActions(dataset_format,img_seq=True)
    actions=action_reader(in_path)
    binary_ensemble=read_ensemble(nn_dir)
    new_actions=binary_ensemble.get_seq(actions)
    save_actions=utils.actions.read.SaveActions(img_actions=False)
    save_actions(new_actions,out_path)

def read_ensemble(nn_dir):
    preproc=tools.ImgPreproc2D()
    nn_reader=deep.reader.NNReader(preproc)
    nn_paths=os.listdir(str(nn_dir))
    nn_paths=[str(nn_dir)+str(nn_path_i)
                for nn_path_i in nn_paths]    
    for nn_i in nn_paths:
    	print(nn_i)
    nn_models=[ nn_reader(nn_i) 
                for nn_i in nn_paths]
    return BinaryEnsemble(nn_models)

if __name__ == "__main__":
    in_path="../../AArtyk/MSR/time/"   
    nn_dir="../../AArtyk/all_models/"
    out_path="../../AArtyk_exp/binary"
    ensemble_seq(in_path,nn_dir,out_path)
