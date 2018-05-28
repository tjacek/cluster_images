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

class Rescale(object):
    def __init__(self,new_dim=(64,64)):
        self.new_dim=new_dim

    def __call__(self,imgs):
        if(type(imgs)==list):
            return [ self.rescale_img(img_i) for img_i in imgs]
        else:
            return self.rescale_img(imgs)

    def rescale_img(self,img_i):
        print(type(img_i))
        new_img=cv2.resize(img_i,self.new_dim, interpolation = cv2.INTER_CUBIC)
        return new_img#utils.imgs.new_img(img_i,new_img)

class ProjFrames(object):
    def __init__(self,zx=True,smooth=(10,10),default_depth=150.0):
        self.zx= 0 if(zx) else 1
        self.default_depth=default_depth
        self.smooth=smooth

    def __call__(self,action_i):
        print(action_i.name)
        action_seq,z_dim=prepare_seq(action_i.img_seq)
        dim_0=action_seq.shape[self.zx+1]
        dim_1=z_dim+2
        clean_img=CleanImg(dim_0,dim_1)
        def proj_helper(img_i):
            proj_i=clean_img()
            for point, z in np.ndenumerate(img_i):
                if(z!=0):
                    i=point[self.zx]
                    j=int(np.floor(z))
                    proj_i[i][j]=self.default_depth
            if(self.smooth):
                proj_i=self.smooth_img(proj_i)      
            return proj_i
        return [ proj_helper(img_i) for img_i in action_i.img_seq]

    def smooth_img(self,raw_img):
    #    se1 = cv2.getStructuringElement(cv2.MORPH_RECT, self.smooth)
    #    smooth_img = cv2.morphologyEx(raw_img, cv2.MORPH_OPEN, se1)
        true_kern=np.ones(self.smooth)
    #    smooth_img= cv2.erode(raw_img, (3,3), iterations=1)
    #    smooth_img=remove_isol(raw_img)
        smooth_img=cv2.dilate(raw_img, true_kern, iterations=1)
        return smooth_img

class CleanImg(object):
    def __init__(self,x,y):
        self.dims=(x,y)

    def __call__(self):
        return np.zeros(self.dims)
        
def time_frames(action_i):
    n=len(action_i)-1
    time_frames=[]
    for i in range(n):
        time_frames.append([action_i[i],action_i[i+1]])
    return time_frames

def prepare_seq(img_seq,z_dim=None,shift=1.0):
    if(z_dim is None):
        z_dim= float(img_seq[0].shape[0])
    action_array=np.array(img_seq)
    z_max=np.amax(action_array)+2.0*shift
    z_min=np.amin( action_array[action_array!=0])
    z_delta=z_max-z_min
    action_array[action_array!=0]+=shift
    action_array[action_array!=0]-=z_min
    action_array[action_array!=0]/=z_delta
    action_array[action_array!=0]*=z_dim
    return action_array,int(z_dim)

if __name__ == "__main__":
    pipline=inject_pipline(FullPipeline(),dataset_format='cp_dataset')
    in_path="../../Documents/AC1/test"
    out_path="../../Documents/AC1/out"
    pipline(in_path,out_path)
    #time_path="../dataset2a/preproc/basic/time"
    #xz_path="../dataset2a/preproc/basic/xz"
    #yz_path="../dataset2a/preproc/basic/yz"
    #out_path="../dataset2a/preproc/unified"
    #proj_unify(time_path,xz_path,yz_path,out_path,'proj_tmp','basic_dataset')
    #apply_unify=UnifyActions(dataset_format='basic_dataset')
    #apply_unify.append(in_path_x,in_path_y,out_path,norm=[False,False])