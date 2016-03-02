import utils.files as files
import utils.imgs as images
import numpy as np

def read_dataset(dir_path):
    cat_dirs=files.get_dirs(dir_path,True)
    all_images=[]
    y=[]
    for i,cat_dir_i in enumerate(cat_dirs):
        images_cat_i=images.read_img_dir(cat_dir_i)
        print(len(images_cat_i))
        for img_j in images_cat_i:
            all_images.append(img_j)
            y.append(i)
    return np.array(all_images),np.array(y)