import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np
from sets import Set 
import utils.actions
import utils.actions.tools
import utils.actions.read
import bow

class SelectFeatures(object):
    def __init__(self, feat_indexes):
        self.feat_indexes=Set(feat_indexes)

    def __call__(self,action_i):
        features=action_i.to_series()
        s_features=[ feature_i 
                        for i,feature_i in enumerate(features)
                            if(i in self.feat_indexes) ]
        img_seq=np.array(s_features).T
        print(img_seq.shape)	
        return utils.actions.tools.new_action(action_i,img_seq)

def select_features(in_path,out_path,dataset_format='cp_dataset'):
    action_reader=utils.actions.read.ReadActions(dataset_format,img_seq=False,as_dict=False)
    actions=action_reader(in_path)
    quality=bow.compute_quality(actions)
    selector=SelectFeatures(get_indexes(quality,30))
    new_actions=[ selector(action_i) for action_i in actions]
    save_actions=utils.actions.read.SaveActions(img_actions=False)
    save_actions(new_actions,out_path)

def get_indexes(quality,k):
    indexes=np.argsort(quality)
    return indexes[-k:]


if __name__ == "__main__":
    in_path="../../AA_disk3/clust_seqs/nn_"
    out_path="../../AA_disk3/s_seqs/nn_"	
    for i in range(20):
    	in_i=in_path+str(i)
    	out_i=out_path+str(i)
        select_features(in_i,out_i)