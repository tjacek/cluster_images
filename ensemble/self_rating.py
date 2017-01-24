import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np
import ensemble.feat_seq

class HighRatedFrames(object):
    def __init__(self,criterion,threshold=0.9):
        self.criterion=criterion
        self.threshold=threshold

    def __call__(self,frames): 
        return [frame_i for frame_i in frames
                     if self.criterion(frame_i)>self.threshold]

class ConvnetCriterion(object):
    def __init__(self, convnet):
        self.convnet=convnet

    def __call__(self,frame_x):
        dist=self.convnet.get_distribution(frame_x)	
        return np.linalg.norm(dist, ord=2)

def make_high_rated_frames(nn_path,threshold=0.9,prep_type="time"):
    conv=ensemble.feat_seq.read_convnet(nn_path,prep_type)
    crit=ConvnetCriterion(conv)
    return HighRatedFrames(crit,threshold)

if __name__ == "__main__":
    nn_path="../dataset1/exp1/nn_trivial"
    make_high_rated_frames(nn_path)