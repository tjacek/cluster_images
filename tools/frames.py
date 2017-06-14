import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import utils.actions
import utils.actions.frames
import utils.actions.unify

def make_time_frames(in_path,out_path,dataset_format='cp_dataset'):
    time_frames=utils.actions.frames.TimeFrames() 
    utils.actions.transform_actions(in_path,out_path,time_frames,
    	                            seq_transform=True,dataset_format=dataset_format)

def make_proj_frames(in_path,out_path,dataset_format='cp_dataset'):
    time_frames=utils.actions.frames.ProjFrames(zx=True,smooth=True) 
    utils.actions.transform_actions(in_path,out_path,time_frames,
    	                            seq_transform=True,dataset_format=dataset_format)

def make_full_frames(time_path,xz_path,out_path,dataset_format='cp_dataset'):
    apply_unify=utils.actions.unify.UnifyActions(dataset_format=dataset_format)
    apply_unify.append(time_path,xz_path,out_path)#,norm=[False,False])

if __name__ == "__main__":
    time_path= "../../exper/time/full"
    xz_path="../../exper/proj_xz"
    out_path="../../exper/unified/full"
    make_full_frames(time_path,xz_path,out_path)