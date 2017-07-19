import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np
import utils.actions
import utils.actions.read

class FrameDesc(object):
    def __init__(self,action,cat,person,index,value):
        
        self.action=  action#self.clean(action)#action
        self.cat=cat
        self.person=person
        self.index=index
        self.value=value
    
    def __len__(self):
        return len(self.value)

    def __str__(self):
        return self.action+"_"+str(self.index)

    def new(self,new_value):
        return FrameDesc(self.action,self.cat,self.person,self.index,new_value)

def merge_features(paths,dataset_format='cp_dataset'):
    frames_desc=[ parse_seq(path_i,dataset_format) 
                  for path_i in paths]
    frame_ids=frames_desc[0].keys()
    def merged_desc(frame_id):
        print(frame_id)
        frames=[ desc_i.get(frame_id,None)
                 for desc_i in frames_desc]
        for frame_i in frames:
            if(frame_i==None):
                return None
        np_values=[ frame_i.value
                      for frame_i in frames]
        merged_values=np.concatenate(np_values)
        return frames[0].new(merged_values)

    merged_frames=[ merged_desc(frame_j)
                     for frame_j in frame_ids]

    merged_frames=[ mf for mf in merged_frames
                       if (mf!=None)]
    #merged_frames=clean_frames(merged_frames)
    return merged_frames 

def clean_frames(merged_frames):
    new_dim=max([ len(mf) for mf in merged_frames])  
    clean_frames=[ mf 
                   for mf in merged_frames
                     if len(mf)==new_dim]
    return merged_frames

def parse_seq(path_i,dataset_format='basic_dataset'):
    read_actions=utils.actions.read.ReadActions(dataset_format=dataset_format,img_seq=False)
    actions=read_actions(path_i)
    frame_desc=[]
    for action_i in actions:
    	frame_desc+=to_frame_descs(action_i)
    return { str(desc_i):desc_i 
               for desc_i in frame_desc}

def to_frame_descs(action_i):
	return [ FrameDesc(action_i.name,action_i.cat,action_i.person,j,frame_j)
	           for j,frame_j in enumerate(action_i.img_seq)]

def actions_from_descs(frames_desc):
    actions={}
    for frame_i in frames_desc:
        action_i=actions.get(frame_i.action,[])
        action_i.append(frame_i)
        actions[frame_i.action]=action_i
    actions=[ parse_action(action_name_i,desc_i) 
              for action_name_i,desc_i in actions.items()] 
    return actions

def parse_action(action_name,frames_desc):
    #action_name=frames_desc[0].name
    unordered_values={desc_i.index:desc_i#.value
                        for desc_i in frames_desc}
    n_desc=len(frames_desc)
    print(unordered_values.keys())
    ordered_values=[ unordered_values[i] 
                     for i in range(n_desc)]
    return (action_name,ordered_values)#utils.action.Action(name,ordered_values)

def save_actions(raw_actions,out_path,dataset_format='cp_dataset'):
    #parse_action_name=utils.actions.read.FORMAT_DIR[dataset_format]
    def parse_action(action_name_i,seq_i):
        #name,cat,person=parse_action_name(action_name_i)
        name=action_name_i
        cat=seq_i[0].cat
        person=seq_i[0].person
        img_seq=[ desc_i.value 
                  for desc_i in seq_i]
        return utils.actions.Action(name,img_seq,cat=cat,person=person)
    actions=[ parse_action(action_name_i,seq_i)
              for action_name_i,seq_i in raw_actions]
    for action_i in actions:
        print(action_i.name)
    save_actions=utils.actions.read.SaveActions(img_actions=False)#,dataset_format=dataset_format)
    save_actions(actions,out_path)

if __name__ == "__main__":
    path1="../../konf2/exp2/seq"
    path2="../../konf2/exp3/seq"
    #path3="../../konf/full/seq"
    out_path='../../konf2/u_seq'
    paths=[path1,path2]
    frames_desc=merge_features(paths,dataset_format='basic_dataset')
    raw_actions=actions_from_descs(frames_desc)
    #print(len(raw_actions))

    save_actions(raw_actions,out_path,dataset_format='basic_dataset')