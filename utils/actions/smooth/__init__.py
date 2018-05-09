import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np 
import utils.actions.read
import utils.actions.tools
import utils.paths.dirs

class TimeSeriesTransform(object):
    def __init__(self,dataset_format='cp_dataset'):
        self.dataset_format=dataset_format

    def __call__(self,in_path,out_path):
        action_reader=utils.actions.read.ReadActions(self.dataset_format,img_seq=False,as_dict=False)
        actions=action_reader(in_path)
        frames=utils.actions.tools.get_frames(actions)
        unit_norm=self.get_series_transform(frames)
        norm_actions=[ action_i.transform(unit_norm, img_seq=False) 
                        for action_i in actions]
        save_actions=utils.actions.read.SaveActions(img_actions=False)
        save_actions(norm_actions,out_path)

    def get_series_transform(self,frames):
        raise NotImplementedError()    	

