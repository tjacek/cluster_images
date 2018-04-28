import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np 
import utils.actions.read
import utils.paths.dirs

class TimeSeriesTransform(object):
    def __init__(self,dataset_format='cp_dataset'):
        self.dataset_format=dataset_format
 
    @utils.paths.dirs.ApplyToFiles(True)
    def __call__(in_path,out_path,dataset_format='cp_dataset'):
        action_reader=utils.actions.read.ReadActions(self.dataset_format,img_seq=False,as_dict=False)
        actions=action_reader(in_path)
        frames=get_frames(actions)
        unit_norm=self.get_series_transform(frames)
        norm_actions=[ action_i.transform(unit_norm, img_seq=False) 
                        for action_i in actions]
        save_actions=utils.actions.read.SaveActions(img_actions=False)
        save_actions(norm_actions,out_path)

    def get_series_transform(self,frames):
        raise NotImplementedError()    	

def get_frames(actions):
    all_frames=[]
    for action_i in actions:
        all_frames+=action_i.img_seq
    return all_frames