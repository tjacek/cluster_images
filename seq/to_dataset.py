import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np 
from sets import Set
import seq
import utils
import utils.dirs as dirs
import utils.files as files
import utils.actions
import utils.data as data
import utils.text #as

def seq_dataset(in_path):
    all_paths=dirs.all_files(in_path)
    seqs=[ parse_seq(path_i)
           for path_i in all_paths]
    get_cat=data.ExtractCat(data.id_cat)
    y=[ get_cat(seq_i.cat) 
        for seq_i in seqs]
    x=[ seq_i.img_seq
        for seq_i in seqs]
    persons=[ seq_i.person
              for seq_i in seqs]
    names=[ seq_i.name
            for seq_i in seqs]          
    dataset=data.make_dataset(x,y,persons)
    dataset['names']=names
    return dataset

def parse_seq(path_i):
    name=path_i.get_name()
    cat=path_i[-2]
    person=utils.text.get_person(name)
    lines=files.read_file(str(path_i))
    data=parse_text(lines)
    return utils.actions.Action(name,data,cat,person)

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

def masked_dataset(dataset):
    x=dataset['x']
    y=dataset['y']
    params= data.make_params(x,y)
    print(params)
    mask=make_mask(x,params['n_batch'],params['max_seq'])
    x_masked=make_masked_seq(x,params['max_seq'],params['seq_dim'])
    new_dataset={'x':x_masked,'y':dataset['y'],'mask':mask,
                 'persons':dataset['persons'],'params':params}
    return new_dataset

def make_mask(x,n_batch,max_seq):
    mask = np.zeros((n_batch, max_seq),dtype=float)
    for i,seq_i in enumerate(x):
        seq_i_len=data.seq_len(seq_i)
        mask[i][:seq_i_len]=1.0
    return mask

def make_masked_seq(x,max_seq,seq_dim):
    def masked_seq(seq_i):
        seq_i_len=data.seq_len(seq_i)
        new_seq_i=np.zeros((max_seq,seq_dim))
        new_seq_i[:seq_i_len]=seq_i[:seq_i_len]
        return new_seq_i
    return [masked_seq(seq_i) for seq_i in x]
       
if __name__ == "__main__":
    path='../dataset0/seq/'
    seq_dataset(path)