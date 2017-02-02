import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import ensemble
from ensemble.single_cls import make_single_cls
import utils.actions
import numpy as np

class MultiNNEnsemble(object):
    def __init__(self,datasets,nnetworks,dispersion=False,vote=False):
        self.datasets=datasets
        self.nnetworks=nnetworks
        self.dispersion=dispersion
        self.vote=vote

    def get_category(self,action):
        result=self(action,self.dispersion,self.vote) 
        print(result)
        result=np.array(result)
        dist=np.sum(result,axis=0)
        print(dist.shape)
        return np.argmax(dist)

    def __call__(self,action, dispersion=False,vote=False):
        distributions=[ self.get_distribution(action,type_i,act_dict_i)
                         for type_i,act_dict_i in self.datasets.items()]
        if(vote):
            return [ get_binary_dist(dist_i) 
                      for dist_i in distributions]
        if(dispersion):
            return [ np.linalg.norm(dist_i,ord=2) * dist_i 
                      for dist_i in distributions]
        else:    
            return distributions    
    
    def get_distribution(self,action_x,type_i,action_dict_i):
        typed_action=action_dict_i[action_x.name]
        print(type(typed_action))
        print(type_i)
        print(action_x.img_seq[0].shape)
        typeed_nn=self.nnetworks[type_i]
        return typeed_nn(typed_action)

def get_binary_dist(dist):
    index=np.argmax(dist)
    binary_dist= np.zeros(dist.shape)
    binary_dist[index]=1.0
    return binary_dist  

def make_multi_nn(dataset_paths,nn_paths):
    if(type(dataset_paths)!=dict):
        raise("type:dataset_paths dict required")
    if(type(nn_paths)!=dict):
        raise("type:nn_paths dict required")
    datasets=make_datasets(dataset_paths)
    nnetworks={type_i:make_single_cls(conv_path=path_pair_i[0],lstm_path=path_pair_i[1],prep_type=type_i)
                  for type_i,path_pair_i in nn_paths.items()}
    return MultiNNEnsemble(datasets,nnetworks)

def make_datasets(dataset_paths,s_action='odd'):
    return {type_i:make_actions_dict(path_i,s_action)
                for type_i,path_i in dataset_paths.items()}

def make_actions_dict(path_i,s_action='odd'):
    actions=ensemble.read_actions(path_i,action_selection=s_action)
    return { action_i.name:action_i
               for action_i in actions}

if __name__ == "__main__":
    dataset_paths={'time':'../dataset1/exp1/full_dataset',
                   'proj':'../dataset1/exp2/cats'}
    nn_paths={'time':('../dataset1/exp1/nn_data_1' ,'../dataset1/exp1/lstm_self'),
              'proj':('../dataset1/exp2/nn_worst' ,'../dataset1/exp2/lstm_worst')}
    ens=make_multi_nn(dataset_paths,nn_paths)

    in_path="../dataset1/exp1/full_dataset"
    read_actions= utils.actions.ReadActions( utils.actions.cp_dataset,norm=True)
    actions=read_actions(in_path)
    s_actions= utils.actions.select_actions(actions)
    ensemble.check_model(ens,s_actions)