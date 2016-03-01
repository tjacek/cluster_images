import files
import cv2

class Action(object):
    def __init__(self,name,frames):
        self.name=name
        self.frames=frames

    def __str__(self):
    	return self.name

    def __getitem__(self,index):
        return self.frames[index]

    def __len__(self):
        return len(self.frames)

def read_action(action_path):
    print(action_path)
    frame_paths=files.get_files(action_path,True)
    frames=[ cv2.imread(frame_path_i) for frame_path_i in frame_paths]
    name=files.get_name(action_path)
    print(frame_paths)
    return Action(name,frames)

