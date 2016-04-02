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

def pairs_to_dataset(pairs):
    X=[pair_i[0] for pair_i in pairs]
    y=[pair_i[1] for pair_i in pairs]
    X=np.array(X,dtype=float)
    y=np.array(to_ints(y),dtype=float)
    return X,y

def to_ints(y):
    k=0
    index_dir={}
    for y_i in y:
        if(not (y_i in index_dir)):
            index_dir[y_i]=k
            k+=1
    print(index_dir.keys())
    int_labels=[index_dir[y_i] for y_i in y]
    return int_labels

def to_vectors(y):
    n_cats=get_n_cats(y)
    y_vec=np.zeros((len(y),n_cats),dtype=float)   
    for i,y_i in enumerate(y):
        y_vec[i][y_i]=1
    return y_vec

def get_n_cats(y):
    return np.amax(y)+1