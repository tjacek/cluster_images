import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import utils
import utils.dirs as dirs
import utils.files as files
import seq
import numpy as np 

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
    x=[ parse_text(seq_i[1]).shape 
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

if __name__ == "__main__":
    path='../dataset0/seq/'
    seq_dataset(path)