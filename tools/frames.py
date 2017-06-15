import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import utils.actions
import utils.actions.frames
import utils.actions.unify

def bound_frames(in_path,out_path,dataset_format='cp_dataset'):
    bound_frames=utils.actions.frames.BoundFrames(True,None,smooth_img=False) 
    utils.actions.transform_actions(in_path,out_path,bound_frames,
                                    seq_transform=True,dataset_format=dataset_format)

def rescale_frames(in_path,out_path,dataset_format='cp_dataset'):
    rescale=utils.actions.unify.Rescale()
    utils.actions.transform_actions(in_path,out_path,rescale,
                                    seq_transform=False,dataset_format=dataset_format)

def make_time_frames(in_path,out_path,dataset_format='cp_dataset'):
    time_frames=utils.actions.frames.TimeFrames() 
    utils.actions.transform_actions(in_path,out_path,time_frames,
    	                            seq_transform=True,dataset_format=dataset_format)

def make_proj_frames(in_path,out_path,dataset_format='cp_dataset'):
    time_frames=utils.actions.frames.ProjFrames(zx=True,smooth=False) 
    utils.actions.transform_actions(in_path,out_path,time_frames,
    	                            seq_transform=True,dataset_format=dataset_format)

def make_motion_frames(in_path,out_path,dataset_format='cp_dataset'):
    motion_frames=utils.actions.frames.MotionFrames(tau=3.0) 
    utils.actions.transform_actions(in_path,out_path,motion_frames,
                                    seq_transform=True,dataset_format=dataset_format)

def make_full_frames(time_path,xz_path,out_path,dataset_format='cp_dataset'):
    apply_unify=utils.actions.unify.UnifyActions(dataset_format=dataset_format)
    apply_unify.append(time_path,xz_path,out_path)#,norm=[False,False])

if __name__ == "__main__":
    raw_path= "../../MSRA/raw_"
    bound_path =  "../../MSRA/bound"
    scale_path = "../../MSRA/scale"
    time_path = "../../MSRA/time"
    proj_path=  "../../MSRA/proj_zx"
    motion_path=  "../../MSRA/motion"
    out_path=     "../../MSRA/full"
    #bound_frames(raw_path,bound_path)
    #rescale_frames(bound_path,scale_path)
    #make_time_frames(scale_path,time_path)
    #make_proj_frames(scale_path,proj_path)
    #make_motion_frames(proj_path,motion_path)
    make_full_frames(time_path,motion_path,out_path)