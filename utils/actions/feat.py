import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np
import utils.actions
import utils.actions.read
from sklearn.svm import SVC
from sklearn.feature_selection import RFE

@utils.paths.dirs.ApplyToFiles(True)
def reduce_actions(in_path,out_path,dataset_format='cp_dataset'):
    action_reader=utils.actions.read.ReadActions(dataset_format,img_seq=False,as_dict=False)
    actions=action_reader(in_path)
    pairs=get_pairs(actions)
    
    rfe=make_transform(pairs)  
    def redu_transform(x):
    	x=x.reshape(1, -1)
        new_frame=rfe.transform(x)
        return new_frame.flatten()

    norm_actions=[ action_i.transform(redu_transform, img_seq=False) 
                    for action_i in actions]
    save_actions=utils.actions.read.SaveActions(img_actions=False)
    save_actions(norm_actions,out_path)

def make_transform(pairs,n=70):
    y=[ pair_i[0] for pair_i in pairs]
    X=[ pair_i[1] for pair_i in pairs]
    svc = SVC(kernel='linear',C=1)
    rfe = RFE(estimator=svc,n_features_to_select=n,step=1)
    rfe.fit(X,y)
    return rfe

def get_pairs(actions):
    all_frames=[]
    for action_i in actions:
        all_frames+=action_i.to_pairs()
    return all_frames

if __name__ == "__main__":
    in_path="../../AA_konf3/basic_deep/"#nn_0"
    out_path="../../AA_konf3/selected/"
    reduce_actions(in_path,out_path)