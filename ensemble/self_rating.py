import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np
import ensemble.feat_seq
import utils.actions 
import utils
class HighRatedFrames(object):
    def __init__(self,criterion,threshold=0.9):
        self.criterion=criterion
        self.threshold=threshold

    def __call__(self,action):
        frames=action.img_seq
        return [frame_i for frame_i in frames
                     if self.criterion(frame_i,action.cat)>self.threshold]


class CatCriterion(object):
    def __init__(self, convnet, cats_ids=None):
        self.convnet=convnet
        self.COUNTER=0
        self.cats_ids=cats_ids

    def __call__(self,frame_x,correct_cat):
    	frame_x.name=str(frame_x.name)
    	img4D=self.convnet.preproc.apply(frame_x)
        
        correct_cat=int(correct_cat)#-1
        cat=self.convnet.get_category(img4D)[0]	
        if(self.cats_ids!=None):
            correct_cat=self.cats_ids[correct_cat]
        print("@@@@@@@@@@@@@@@@@@@@@")
        print(correct_cat)
        print(cat)
        if(correct_cat!=cat):   
            self.COUNTER+=1.0
            print("counter " + str(self.COUNTER))
            return 0.0
        return 1.0

class DispCriterion(object):
    def __init__(self, convnet):
        self.convnet=convnet

    def __call__(self,frame_x,correct_cat):
        dist=self.convnet.get_distribution(frame_x)#[0]	
        print(dist)
        #value=np.linalg.norm(dist, ord=2)
        #print(value)
        return 1.0#value

def make_high_rated_frames(nn_path,threshold=0.9,prep_type="time", cats_ids=None):
    conv=ensemble.feat_seq.read_convnet(nn_path,prep_type)
    crit=CatCriterion(conv,cats_ids)
    #crit=DispCriterion(conv)
    return HighRatedFrames(crit,threshold)

if __name__ == "__main__":
    nn_path="../dataset1/exp1b/old/nn_trivial"
    dataset_path="../dataset1/exp1b/train_trivial"
    output_path="../dataset1/exp1b/train_self"
    trivial={4:0,9:1,10:2,11:3,12:4,13:5,14:6,15:7,16:8}
    high_rated=make_high_rated_frames(nn_path,prep_type='time',cats_ids=trivial)
    
    #x,y=imgs.to_dataset(imgset,extract_cat,preproc)
    
    actions=ensemble.feat_seq.read_actions(dataset_path)
    for action_i in actions:
        print(action_i.cat)
    new_actions=[ action_i(high_rated) 
                 for action_i in actions]
    new_actions=[ action_i 
                  for action_i in new_actions
                    if len(action_i) >0 ]
    for action_i in new_actions:
        print(len(action_i))                
    utils.actions.save_actions(new_actions,output_path)                
    print(len(new_actions))
