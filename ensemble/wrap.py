import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np
import ensemble.feat_seq
from seq.dtw import dtw_metric

class SingleDTW(object):
    def __init__(self,conv,actions,cats):
        self.actions=actions
        self.cats=cats
        self.conv=conv

    def __call__(self,action,k=10):
        action_feat=self.conv(action)
        distance=[dtw_metric(action_feat,action_i) 
                    for action_i in self.actions]
        distance=np.array(distance)
        dist_inds=distance.argsort()[0:k]
        distance_k=distance[dist_inds]
        print(distance_k.shape)
        cats_k =[ self.cats[i] for i in dist_inds]
        return cats_k,distance_k

    def dataset_scale(self):
        pair_dist=[dtw_metric(action_i,action_j)
            for action_i in self.actions
              for action_j in self.actions]
        return np.array(pair_dist).mean()

def make_single_dtw(nn_path,dataset_path,prep_type="time"):
    conv=ensemble.feat_seq.make_feat_seq(nn_path,prep_type)
    actions=ensemble.feat_seq.read_actions(dataset_path)
    conv_actions=[conv(action_i) 
                   for action_i in actions]
    cats=[ action_i.cat
            for action_i in actions]
    return SingleDTW(conv,conv_actions,cats)

if __name__ == "__main__":
    nn_path="../dataset1/exp1/nn_full"
    dataset_path="../dataset1/exp1/train_full"
    single_dtw=make_single_dtw(nn_path,dataset_path)
    actions=ensemble.feat_seq.read_actions(dataset_path)
    print(single_dtw(actions[0]))
    #print(single_dtw.dataset_scale())