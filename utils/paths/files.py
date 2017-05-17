import os
import os.path as io 
import pickle,re 
from natsort import natsorted
from shutil import copyfile
import numpy as np

def dir_to_txt(in_path,out_path):
    dir_content=get_files(in_path)
    dir_content=append_path(in_path,dir_content)
    text="\n".join(dir_content)
    save_string(out_path,text)

def read_file(path,lines=True):
    file_object = open(str(path),'r')
    if(lines):
        text=file_object.readlines()  
    else:
        text=file_object.read()
    file_object.close()
    return text

def array_to_txt(array,sep=""):
    return sep.join(array)

def dict_to_txt(text_dict,sep='#'):
    lines=[ str(key_i)+ sep + vector_to_string(value_i)
            for key_i,value_i in text_dict.items()]
    return '\n'.join(lines) 

def txt_to_dict(text,sep='#'):
    lines=text.split('\n')
    pairs=[ line_i.split(sep) 
               for line_i in lines]
    vec_dict=dict([ (pair_i[0], string_to_vector(pair_i[1]))
                      for pair_i in pairs
                        if len(pair_i)==2])
    return vec_dict 

def save_object(nn,path):
    file_object = open(path,'wb')
    pickle.dump(nn,file_object)
    file_object.close()

def save_array(array,out_path):
    txt=array_to_txt(array,"\n")
    save_string(out_path,txt)

def save_string(path,string):
    file_str = open(str(path),'w')
    file_str.write(string)
    file_str.close()

def read_object(path):
    file_object = open(path,'r')
    obj=pickle.load(file_object)  
    file_object.close()
    return obj

def seq_to_string(seq,extra_data=None):
    if(exta_data!=None):
        array=[ vector_to_string(elem_i) + extra_data(i)
                 for i,elem_i in enumerate(seq)]    
    else:
        array=[vector_to_string(elem_i) 
                 for elem_i in seq]
    return '\n'.join(array)

def vector_to_string(vec):
    array=[str(vec_i) for vec_i in vec]
    return ','.join(array)

def string_to_vector(text):
    raw_vec=text.split(',')
    vec=[float(vec_i) for vec_i in raw_vec]
    return np.array(vec)

def extract_prefix(filename):
    return filename.split(".")[-1]
