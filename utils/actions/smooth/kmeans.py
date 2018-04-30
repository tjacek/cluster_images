import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np 
import utils.actions.smooth

class ClsDiskr(object):
    def __init__(self,var):
        self.k_means=var

    def __call__(self,x):
        def cls_helper(i,x_i):
            return np.argmax(self.k_means[i])
        return [ cls_helper(i,x_i) for i,x_i in enumerate(x)]