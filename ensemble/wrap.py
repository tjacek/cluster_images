import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np
import ensemble.feat_seq
from seq.dtw import dtw_metric

class SingleDTW(object):
    def __init__(self,conv,actions,k=1):
        self.actions=actions
        self.k=k
        self.conv=conv

    def __call__(self,action):
        action_feat=self.conv(action)
        distance=[self.metric(x,action_i) 
                    for action_i in self.actions]
        distance=np.array(distance)
        dist_inds=distance.argsort()[0:k]

    def dataset_scale(self):
        pair_dist=[dtw_metric(action_i,action_j)
            for action_i in self.actions[0:2]
              for action_j in self.actions[0:2]]
        print(len(pair_dist))
        return np.array(pair_dist).mean()

def make_single_dtw(nn_path,dataset_path,prep_type="time",k=1):
    conv=ensemble.feat_seq.make_feat_seq(nn_path,prep_type)
    actions=ensemble.feat_seq.read_actions(dataset_path)
    conv_actions=[conv(action_i) 
                   for action_i in actions]
    return SingleDTW(conv,conv_actions,k)

if __name__ == "__main__":
    nn_path="../dataset1/exp1/nn_trivial"
    dataset_path="../dataset1/exp1/train_full"
    single_dtw=make_single_dtw(nn_path,dataset_path)
    print(single_dtw.dataset_scale())