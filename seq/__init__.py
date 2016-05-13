import utils
import utils.files as files
import numpy as np
import to_dataset


ALPH="?ABCDEFGHIJKLMN"

class Seq(object):
    def __init__(self,items,cat,name):
        self.seq=items
        self.cat=cat
        self.name=name
        
    def __getitem__(self,index):
        return self.seq[index]

    def __str__(self): 
        return self.name +" "+ str(len(self.seq))+" "+str(self.cat)

    def __len__(self):
        return len(self.seq)

def create_seqs(actions,out_path):
    seqs=[action_to_seq(action_i) for action_i in actions]
    seq_txt=files.array_to_txt(seqs,sep="\n")
    files.save_string(out_path,seq_txt)

def action_to_vec_seq(action,n_cats=10):
    return [get_vector_cat(symb_i,n_cats) for symb_i in action.seq]

def get_vector_seq(str_seq_i,n_cats=10):
    return [ get_vector_cat(symb_i,n_cats) for symb_i in str_seq_i]

def get_vector_cat(symbol,n_cats=10):
    vec=np.zeros((n_cats,))
    vec[get_cat(symbol)]=1
    return vec

def get_cat(symbol):
    print(symbol)
    return ALPH.index(symbol)

def get_seqs(str_seqs):
    inst=get_instances(str_seqs)
    cats=get_classes(str_seqs)
    return [Seq(inst_i,cat_i) for inst_i,cat_i in zip(inst,cats)]

def get_instances(str_seqs):
    cats=[symb_i.split("#")[0] for symb_i in str_seqs]
    cats=[ get_vector_seq(str_seq_i) for str_seq_i in cats]
    return cats

def get_classes(str_seqs):
    cat_names={} 
    classes=[  symb_i.split("#")[1] for symb_i in str_seqs]
    classes=[utils.get_value(cl_i,cat_names) for cl_i in classes]
    return classes