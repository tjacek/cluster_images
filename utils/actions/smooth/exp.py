import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np 
import utils.actions.smooth
import utils.actions.tools
import utils.actions.smooth.kmeans

class ExpSmooth(utils.actions.smooth.TimeSeriesTransform):
    def __init__(self,alpha=0.5,dataset_format='cp_dataset'):
        super(ExpSmooth, self).__init__(features=True,dataset_format=dataset_format)
        self.alpha=alpha

    def get_series_transform(self,frames):
        return exp

def exp(cords,alpha=0.3):
    new_cord=[cords[0]]
    beta=1.0-alpha
    for cord_i in cords[1:]:
        new_cord.append(beta*cord_i + alpha*new_cord[-1])
    return new_cord
            
if __name__ == "__main__":
    in_path="../../Documents/AA/norm_seqs/"
    out_path="../../Documents/AA/smooth_seqs/"
    clust_disk=ExpSmooth()
    path_dec=utils.paths.dirs.ApplyToFiles(True)
    clust_disk= path_dec(clust_disk)
    clust_disk(in_path,out_path)