import utils
import scipy.misc as image
from instances import Instance

def read_images(path):
    action_files=utils.get_dirs(path)
    action_files=utils.append_path(path,action_files)
    images=[]
    for action_path in action_files:
        images+=read_action(action_path)
    return instances.Dataset(images)

def read_action(action_path):
    all_files=utils.get_files(action_path)
    all_files=utils.append_path(action_path+"/",all_files)
    instances=[]
    for full_path in all_files:
        img=image.imread(full_path)
        img=img.flatten()
        img=img.astype(float)
        img/=sum(img)
        instances.append(Instance(full_path,img))
    return instances

def get_number_of_batches(batch_size,dataset):
    n_images=len(dataset)
    n_batches=n_images / batch_size
    if(n_images % batch_size != 0):
        n_batches+=1
    return n_batches
