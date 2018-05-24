import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np 
import utils.actions.read
from utils.actions.tools import count_feats,get_frames

class ActionStatistic(object):
    def __init__(self,global_stat,local_stat=None,str_format='stats %d'):
        self.local_stat=local_stat
        self.global_stat=global_stat
        self.str_form=str_format

    def __call__(self,actions,as_str=False):
        if(self.local_stat is None):
            stat=self.global_stat(actions)
        else:
            stat=self.global_stat( [ self.local_stat(action_i) 
                                    for action_i in actions])
        if(as_str):
            return (self.str_form % stat)
        else:
            return stat      

def show_stats(actions):
    #action_statistics=[ ActionStatistic(max,len,name='avg of frames'),
    #                    ActionStatistic(min,len,name='min of frames'),
    #                    ActionStatistic(np.median,len,name='median of len'),
    #                    ActionStatistic(np.average,len,name='avg')]
    basic_stats=[ ActionStatistic(len,None,str_format='Number of actions %d'),
                  ActionStatistic(count_feats,None,str_format='Number of features %d'),
                  ActionStatistic(sum,len,str_format='Number of frames %d'),
                  ActionStatistic(is_normal,None,str_format='Var %d %d')]
    for stat_i in basic_stats:
        print(stat_i(actions,as_str=True))

def is_normal(actions):
    frames=get_frames(actions)
    means=np.mean(frames,axis=0)
    stds=np.std(frames,axis=0)
    return np.amax(means),np.amax(stds)

if __name__ == "__main__": 
    #action_path="../dataset2/exp1/cats"
    action_path="../dataset2a/exp3/seq"
    read_actions=utils.actions.read.ReadActions( img_seq=False, 
                        dataset_format='basic_dataset')
    actions=read_actions(action_path)
    show_stats(actions)