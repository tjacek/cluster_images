import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import ensemble
from ensemble.single_cls import make_single_cls

class MultiNNEnsemble(object):
    def __init__(self,datasets,nnetworks):
        self.datasets=datasets
        self.nnetworks=nnetworks

def make_multi_nn(dataset_paths,nn_paths):
    if(type(dataset_paths)!=dict):
        raise("type:dataset_paths dict required")
    if(type(nn_paths)!=dict):
        raise("type:nn_paths dict required")
    datasets={type_i:ensemble.read_actions(path_i)
                for type_i,path_i in dataset_paths.items()}
    nnetworks={type_i:make_single_cls(conv_path=path_pair_i[0],lstm_path=path_pair_i[1],prep_type=type_i)
                  for type_i,path_pair_i in nn_paths.items()}
    return MultiNNEnsemble(datasets,nnetworks)


if __name__ == "__main__":
    dataset_paths={'time':'../dataset1/exp1/full_dataset',
                   'proj':'../dataset1/exp2/cats'}
    nn_paths={'time':('../dataset1/exp1/self_nn' ,'../dataset1/exp1/lstm_self'),
              'proj':('../dataset1/exp2/nn_self' ,'../dataset1/exp2/lstm_self')}
    make_multi_nn(dataset_paths,nn_paths)