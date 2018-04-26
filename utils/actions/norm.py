import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np 
import utils.actions.read
import utils.paths.dirs

class UnitNormalization(object):
    def __init__(self,mean,var):
        self.mean=mean
        self.var=var

    def __call__(self,x):
        def norm_helper(i,x_i):
            if(self.var[i]==0):
                return 0
            return  (x_i - self.mean[i])/self.var[i]
        return [ norm_helper(i,x_i) for i,x_i in enumerate(x)]

class Binarize(object):
    def __init__(self,mean,var):
        self.mean=mean
        self.var=var

    def __call__(self,x):
        def bin_helper(i,x_i):
            if(self.var[i]==0):
                return 0
            if(x_i==0):
                return 0.0
            k=int((x_i - self.mean[i])/self.var[i])
            return k
        return [ bin_helper(i,x_i) for i,x_i in enumerate(x)]        

class SimpleBinarize(object):
    def __call__(self,x):
        def bin_helper(i,x_i):
            if(x_i>0):
                return 1.0
            else:
                return 0.0 
        return [ bin_helper(i,x_i) for i,x_i in enumerate(x)] 


@utils.paths.dirs.ApplyToFiles(True)
def normalize_actions(in_path,out_path,dataset_format='cp_dataset'):
    action_reader=utils.actions.read.ReadActions(dataset_format,img_seq=False,as_dict=False)
    actions=action_reader(in_path)
    frames=get_frames(actions)
    unit_norm=make_unit_normalization(frames)
    norm_actions=[ action_i.transform(unit_norm, img_seq=False) 
                    for action_i in actions]
    save_actions=utils.actions.read.SaveActions(img_actions=False)
    save_actions(norm_actions,out_path)

def make_unit_normalization(frames):
    frames=np.array(frames)
    fr_mean=np.mean(frames,axis=0)
    fr_std= np.std(frames,axis=0)#np.std(frames,axis=0)
    return Binarize(fr_mean,fr_std)

def get_frames(actions):
    all_frames=[]
    for action_i in actions:
        all_frames+=action_i.img_seq
    return all_frames

if __name__ == "__main__":
    in_path="../../AA_disk/all_seqs/"#nn_0"
    out_path="../../AA_disk/disk2_seqs/"
    normalize_actions(in_path,out_path)