import sys,os
sys.path.append(os.path.abspath('../cluster_images'))

class MultiNNEnsemble(object):
    def __init__(self,datasets,nnetworks):
        self.datasets=datasets
        self.nnetworks=nnetworks

def make_multi_nn(dataset_paths,nn_paths):
    if(type(dataset_paths)!=dict):
        raise("type:dataset_paths dict required")
    if(type(nn_paths)!=dict):
        raise("type:nn_paths dict required")
    
if __name__ == "__main__":
    dataset_paths={'time':'../dataset1/exp1/full_dataset',
                   'proj':'../dataset1/exp2/cats'}
    nn_paths={'time':('../dataset1/exp1/self_nn' ,'../dataset1/exp1/lstm_self'),
              'proj':('../dataset1/exp2/nn_self' ,'../dataset1/exp2/lstm_self')}
    make_multi_nn(dataset_paths,nn_paths)