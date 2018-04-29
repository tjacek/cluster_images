import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np 
import utils.actions.smooth

class BinarizeActions(utils.actions.smooth.TimeSeriesTransform):
    
    def get_series_transform(self,frames):
        frames=np.array(frames)
        fr_std= np.std(frames,axis=0)
        return Binarization(fr_std)   

class Binarization(object):
    def __init__(self,var):
        self.var=var

    def __call__(self,x):
        def norm_helper(i,x_i):
            if(self.var[i]==0):
                return 0
            return  int((x_i)/self.var[i])
        return [ norm_helper(i,x_i) for i,x_i in enumerate(x)]