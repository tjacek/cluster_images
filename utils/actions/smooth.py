import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np
import utils.actions
import utils.actions.read
from pandas import Series

def smooth_actions(in_path,out_path,dataset_format='cp_dataset'):
    action_reader=utils.actions.read.ReadActions(dataset_format,img_seq=False,as_dict=False)
    actions=action_reader(in_path)
    smooth_actions=[ smooth_helper(action_i) for action_i in actions]
    save_actions=utils.actions.read.SaveActions(img_actions=False)
    save_actions(smooth_actions,out_path)

def smooth_helper(action_i):
    series_i=action_i.to_series()
    series_i=[Series(serie_ij) 
	            for serie_ij in series_i]
    smooth_series_i=[series_ij.rolling(window=5)
                        for series_ij in series_i]
    smooth=np.array([series_ij.mean().dropna() for series_ij in smooth_series_i])
    n=smooth.shape[1] 
    frames=[smooth[:,i] for i in range(n)]
    return utils.actions.new_action(action_i,frames)

if __name__ == "__main__":
    in_path="../../AA_konf2/basic/"#nn_0"
    out_path="../../AA_konf2/basic_smooth/"
    smooth_actions(in_path,out_path)