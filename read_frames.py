import utils
import scipy.misc as image
from instances import Instance

def read_images(path):
    action_files=utils.get_dirs(path)
    action_files=utils.append_path(path,action_files)
    images=[]
    for action_path in action_files:
        images+=read_action(action_path)
    return images

def read_action(action_path):
    all_files=utils.get_files(action_path)
    all_files=utils.append_path(action_path+"/",all_files)
    instances=[]
    for full_path in all_files:
        img=image.imread(full_path)
        img=img.flatten()
        instances.append(Instance(full_path,img))
    return instances

if __name__ == "__main__":
    path="images/"
    print(read_images(path)[0])

