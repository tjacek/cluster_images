import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np
import deep.reader,deep.tools as tools

class BinaryEnsemble(object):
    def __init__(self,binary_networks):
        self.binary_networks=binary_networks

    def get_seq(self,actions):
        return [action_i.transform(self) 
                    for action_i in self.actions]    	
    
    def __call__(self,img_i):
        feats=[nn_j.get_distribution(img_i)[0] 
                for nn_j in self.binary_networks]
        return np.array(feats)

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
    read_ensemble("../../AArtyk/all_models/")
