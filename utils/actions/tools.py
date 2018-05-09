import numpy as np

def get_frames(actions):
    all_frames=[]
    for action_i in actions:
        all_frames+=action_i.img_seq
    return all_frames

def to_features(frames):
    frames=np.array(frames)
    frames=frames.T    
    return frames.tolist()