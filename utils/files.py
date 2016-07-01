import os
import os.path as io 
import pickle,re 
from natsort import natsorted
from shutil import copyfile

def dir_to_txt(in_path,out_path):
    dir_content=get_files(in_path)
    dir_content=append_path(in_path,dir_content)
    text="\n".join(dir_content)
    save_string(out_path,text)

def read_file(path):
    file_object = open(path,'r')
    lines=file_object.readlines()  
    file_object.close()
    return lines

def array_to_txt(array,sep=""):
    return sep.join(array)

def save_object(nn,path):
    file_object = open(path,'wb')
    pickle.dump(nn,file_object)
    file_object.close()

def save_array(array,out_path):
    txt=array_to_txt(array,"\n")
    save_string(out_path,txt)

def save_string(path,string):
    file_str = open(path,'w')
    file_str.write(string)
    file_str.close()

def read_object(path):
    file_object = open(path,'r')
    obj=pickle.load(file_object)  
    file_object.close()
    return obj

def seq_to_string(seq):
    array=[vector_to_string(elem_i) 
            for elem_i in seq]
    return '\n'.join(array)

def vector_to_string(vec):
    array=[str(vec_i) for vec_i in vec]
    return ','.join(array)

def replace_sufix(sufix,files):
    return map(lambda s:s.replace(
        sufix,""),files)

def extract_prefix(filename):
    return filename.split(".")[-1]
