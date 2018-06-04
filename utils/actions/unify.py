import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np 
import cv2
import utils.imgs
import utils.actions
import utils.actions.read
import utils.paths
import utils.paths.dirs
import utils.actions.tools

class UnifyPipeline(object):
    def __init__(self,transforms,preproc):
        self.transforms=transforms
        self.preproc=preproc

    def __call__(self,action_i):
        action_frames=self.get_multi_frames(action_i) 
        img_seq=[ self.unify(frames_i) 
                    for frames_i in action_frames]
        return utils.actions.tools.new_action(action_i,img_seq)

    def get_multi_frames(self,action_i):
        new_frames=[]
        for transform_i in self.transforms:
            new_frames+=[transform_i(action_i)]
        def frame_helper(frames):
            flat_frames=[]
            for frame_i in frames:
                if(type(frame_i)==list):
                    flat_frames+=frame_i
                else:
                    flat_frames.append(frame_i)
            return flat_frames
        return [frame_helper(tuple_i) for tuple_i in zip(*new_frames)]

    def unify(self,frames):      
        proc_frames=self.preproc(frames) #for frame_i in frames] 
        conc_frames=np.concatenate(proc_frames,axis=0)
        return conc_frames

class BasicPipeline(UnifyPipeline):
    def __init__(self):
        transforms=[time_frames]
        preproc=Rescale()
        super(BasicPipeline, self).__init__(transforms,preproc)

class FullPipeline(UnifyPipeline):
    def __init__(self):
        transforms=[time_frames,ProjFrames(zx=True),ProjFrames(zx=False)]
        preproc=Rescale()
        super(FullPipeline, self).__init__(transforms,preproc)
        
def inject_pipline(pipline,dataset_format='cp_dataset'):
    a_tranform=utils.actions.tools.ActionTransform(transform_type='action',in_seq=True,
                                         out_seq=True,dataset_format=dataset_format)
    return lambda in_path,out_path: a_tranform(in_path,out_path,pipline)

if __name__ == "__main__":
    pipline=inject_pipline(FullPipeline(),dataset_format='cp_dataset')
    in_path="../../Documents/AC1/bound"
    out_path="../../Documents/AC1/full"
    pipline(in_path,out_path)