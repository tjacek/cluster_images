import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np
import ensemble.feat_seq
import utils.actions 

class HighRatedFrames(object):
    def __init__(self,criterion,threshold=0.9):
        self.criterion=criterion
        self.threshold=threshold

    def __call__(self,action):
        frames=action.img_seq
        return [frame_i for frame_i in frames
                     if self.criterion(frame_i,action.cat)>self.threshold]


class CatCriterion(object):
    def __init__(self, convnet):
        self.convnet=convnet
        self.COUNTER=0

    def __call__(self,frame_x,correct_cat):
    	frame_x.name=str(frame_x.name)
    	img4D=self.convnet.preproc.apply(frame_x)
        correct_cat=int(correct_cat)-1
        cat=self.convnet.get_category(img4D)[0]	
        if(correct_cat!=cat):
            print(correct_cat)
            print(cat)
            self.COUNTER+=1.0
            print(self.COUNTER)
            return 0.0
        return 1.0

class ConvnetCriterion(object):
    def __init__(self, convnet):
        self.convnet=convnet

    def __call__(self,frame_x):
        dist=self.convnet.get_distribution(frame_x)[0]	
        value=np.linalg.norm(dist, ord=2)
        print(value)
        return value

def make_high_rated_frames(nn_path,threshold=0.9,prep_type="time"):
    conv=ensemble.feat_seq.read_convnet(nn_path,prep_type)
    crit=CatCriterion(conv)
    return HighRatedFrames(crit,threshold)

if __name__ == "__main__":
    nn_path="../dataset1/exp2/nn_full"
    dataset_path="../dataset1/exp2/train_full"
    output_path="../dataset1/exp2/train_self"
    high_rated=make_high_rated_frames(nn_path,prep_type='proj')
    actions=ensemble.feat_seq.read_actions(dataset_path)
    new_actions=[ action_i(high_rated) 
                 for action_i in actions]
    new_actions=[ action_i 
                  for action_i in new_actions
                    if len(action_i) >0 ]
    utils.actions.save_actions(new_actions,output_path)                
    print(len(new_actions))
