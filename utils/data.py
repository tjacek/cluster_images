import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import files #as files
import utils.imgs as images
import utils.actions as action
import utils.paths
import numpy as np

class ExtractCat(object):
    def __init__(self,parse_cat=None):
        self.dir={}
        if(parse_cat==None):
            self.parse_cat=img_cat
        else:
            self.parse_cat=parse_cat

    def names(self):
        return self.dir.keys()

    def __getitem__(self,i):
        if(not i in self.dir):
            self.dir[i]=len(self.dir)
        return self.dir[i]

    def __call__(self,img_path):
        print(img_path)
        str_i=self.parse_cat(img_path)
        return self[str_i]

def OneHot(object):
    def __init__(self,n_cats):
        self.n_cats=n_cats

    def __call__(self,cat_i):
        vec=np.zeros((self.n_cats,))
        vec[cat_i]=1
        return vec

@utils.paths.path_args
def img_cat(img_path):
    str_i=str(img_path[-3])
    return str_i

def make_dataset(x,y):
    params=make_params(x,y)
    return {'x':x,'y':y,'params':params}

def make_params(x,y):
    max_seq = max_length(x)
    n_batch = len(x)
    n_cats=get_n_cats(y)
    seq_dim=x[0].shape[1]
    params={'n_batch':n_batch,'max_seq':max_seq,'seq_dim':seq_dim,'n_cats':n_cats}
    return params

def max_length(x):
    max_len=[ seq_len(x_i) for x_i in x]
    return max(max_len)

def seq_len(seq_i):
    return seq_i.shape[0]

def get_n_cats(y):
    return np.amax(y)+1

def projection(data,d):
    return [data_i[d] for data_i in data]

#if __name__ == "__main__":
#    path='../dataset2/seq/'
#    use_dtw(path)