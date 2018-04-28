import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np 
import utils.actions.smooth

class NormalizeActions(utils.actions.smooth.TimeSeriesTransform):
    
    def get_series_transform(self,frames):
        frames=np.array(frames)
        fr_mean=np.mean(frames,axis=0)
        fr_std= np.std(frames,axis=0)
        return UnitNormalization(fr_mean,fr_std)   

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

def make_unit_normalization(frames):
    frames=np.array(frames)
    fr_mean=np.mean(frames,axis=0)
    fr_std= np.std(frames,axis=0)
    return Binarize(fr_mean,fr_std)