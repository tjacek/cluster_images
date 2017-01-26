import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np
import utils.actions
import ensemble
import ensemble.feat_seq
from seq.dtw import dtw_metric

class EnsembleDTW(object):
    def __init__(self,single_cls):
        self.single_cls=single_cls

    def get_category(self,action_x):
        results=[cls_i(action_x) 
                  for cls_i in self.single_cls]
        cats_k=[ cat_i[0] for cat_i,dist_i in results ]
        dist_k=[ dist_i[0] for cat_i,dist_i in results]          
        #return cats_k,dist_k
        min_i=np.argmin(dist_k)
        return cats_k[min_i]

class SingleDTW(object):
    def __init__(self,conv,actions,cats,scale=1.0):
        self.actions=actions
        self.cats=cats
        self.conv=conv
        self.scale=scale

    def __call__(self,action,k=10):
        action_feat=self.conv(action)
        distance=[dtw_metric(action_feat,action_i) 
                    for action_i in self.actions]
        distance=np.array(distance)
        dist_inds=distance.argsort()[0:k]
        distance_k=distance[dist_inds]
        print(distance_k.dtype)
        distance_k*=self.scale
        #print(distance_k.shape)
        cats_k =[ int(self.cats[i])-1
                  for i in dist_inds]
        print(distance_k)
        print(cats_k)
        return cats_k,distance_k

    def dataset_scale(self):
        pair_dist=[dtw_metric(action_i,action_j)
            for action_i in self.actions
              for action_j in self.actions]
        return np.array(pair_dist).mean()

def make_ensemble_dtw(dataset_path,nn_paths,scales,prep_type="time"):
    single_cls=[make_single_dtw(nn_path_i,dataset_path,scale_i,prep_type) 
                  for nn_path_i,scale_i in zip(nn_paths,scales)]
    return EnsembleDTW(single_cls)                

def make_single_dtw(nn_path,dataset_path,scale,prep_type="time",action_type='even'):
    conv=ensemble.feat_seq.make_feat_seq(nn_path,prep_type)
    actions=ensemble.feat_seq.read_actions(dataset_path)
    actions=utils.actions.select_actions(actions,action_type)
    conv_actions=[conv(action_i) 
                   for action_i in actions]
    cats=[ action_i.cat
            for action_i in actions]
    return SingleDTW(conv,conv_actions,cats,scale)

if __name__ == "__main__":
    nn_paths=["../dataset1/exp1/nn_full","../dataset1/exp1/nn_trivial"]
    scales=[0.3,1.0]
    dataset_path="../dataset1/exp1/full_dataset"
    ensemble_cls=make_ensemble_dtw(dataset_path,nn_paths,scales)
    #single_dtw=make_single_dtw(nn_path,dataset_path)
    #actions=ensemble.feat_seq.read_actions(dataset_path)
    ensemble.test_model(ensemble_cls,dataset_path)
   # print(ensemble_cls(actions[0]))