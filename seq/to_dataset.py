import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np 
from sets import Set
import seq
import utils
import utils.dirs as dirs
import utils.files as files

class IntCat(object):
    def __init__(self):
        self.names2int={}

    def __call__(self,cat_name):
        if(not cat_name in self.names2int):
            self.names2int[cat_name]=len(self.names2int)
        return self.names2int[cat_name] 

def seq_dataset(in_path):
    all_paths=dirs.all_files(in_path)
    seqs=[ parse_seq(path_i)
           for path_i in all_paths]
    get_cat=IntCat()
    y=[ get_cat(seq_i[0]) 
        for seq_i in seqs]
    x=[ parse_text(seq_i[1])
        for seq_i in seqs]    
    return make_dataset(x,y)

def parse_seq(path_i):
    name=path_i[-2]
    lines=files.read_file(str(path_i))
    return name,lines

def parse_text(lines):
    x_i=[line_to_vector(line_i) 
           for line_i in lines]
    return np.array(x_i)

def line_to_vector(line):
    dims=line.split(',')
    vec_i=np.zeros((len(dims),))
    for j,dim_j in enumerate(dims):
        vec_i[j]=float(dim_j)
    return vec_i

def make_dataset(x,y):
    return {'x':x,'y':y}

def masked_dataset(dataset):
    x=dataset['x']
    max_seq = max_length(x)
    n_batch = len(x)
    seq_dim=x[0].shape[1]
    n_cats=get_n_cats(dataset)
    params={'n_batch':n_batch,'max_seq':max_seq,'seq_dim':seq_dim,'n_cats':n_cats}
    mask=make_mask(x,n_batch,max_seq)
    x_masked=make_masked_seq(x,max_seq,seq_dim)
    new_dataset={'x':x_masked,'y':dataset['y'],'mask':mask,'params':params}
    return new_dataset

def max_length(x):
    max_len=[ seq_len(x_i) for x_i in x]
    return max(max_len)

def make_mask(x,n_batch,max_seq):
    mask = np.zeros((n_batch, max_seq),dtype=float)
    for i,seq_i in enumerate(x):
        seq_i_len=seq_len(seq_i)
        mask[i][:seq_i_len]=1.0
    return mask

def make_masked_seq(x,max_seq,seq_dim):
    def masked_seq(seq_i):
        seq_i_len=seq_len(seq_i)
        new_seq_i=np.zeros((max_seq,seq_dim))
        new_seq_i[:seq_i_len]=seq_i[:seq_i_len]
        return new_seq_i
    return [masked_seq(seq_i) for seq_i in x]
       
def seq_len(seq_i):
    return seq_i.shape[0]

def get_n_cats(dataset):
    cats=Set()
    cats.update(dataset['y'])
    return len(cats)

if __name__ == "__main__":
    path='../dataset0/seq/'
    seq_dataset(path)