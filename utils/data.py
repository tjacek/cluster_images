import files #as files
import utils.imgs as images
import utils.paths
import numpy as np

class ExtractCat(object):
    def __init__(self):
        self.dir={}

    def __getitem__(self,i):
        if(not i in self.dir):
            self.dir[i]=len(self.dir)
        return self.dir[i]

    def __call__(self,img_path):
        img_path=utils.paths.Path(img_path) 
        str_i=str(img_path[-3])
        return self[str_i]

def OneHot(object):
    def __init__(self,n_cats):
        self.n_cats=n_cats

    def __call__(self,cat_i):
        vec=np.zeros((self.n_cats,))
        vec[cat_i]=1
        return vec

def get_n_cats(y):
    return np.amax(y)+1

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

def agum_dataset(pairs):
    new_pairs=[]
    for pair_i in pairs:
        org_img=pair_i[0].get_orginal()
        org_img=org_img[::-1]
        new_pairs.append((org_img,pair_i[1]))
    pairs+=new_pairs
    return pairs

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

def extract_cat(X,y,cat):
    print(X.shape)
    X_cat=[X[i] for i,y_i in enumerate(y)
             if y_i==cat]
    return X_cat

def dataset_to_labels(out_path,X,y):
    text=""
    y=to_ints(y)
    for i,y_i in enumerate(y):
        vector=utils.files.vector_string(X[i])
        text+=vector+",#"+str(y_i)+"\n"
    utils.files.save_string(out_path,text)