
import utils
import cv2
import files

def read_images(files):
    imgs=[cv2.imread(f,cv2.IMREAD_GRAYSCALE) for f in files]#[image.imread(f) for f in files]
    return [img_i.reshape((1,3600)) for img_i in imgs
                             if img_i!=None]

def read_img_dir(action_path):
    print(action_path)
    all_files=files.get_files(action_path)
    all_files=files.append_path(action_path+"/",all_files)
    return read_images(all_files) 
