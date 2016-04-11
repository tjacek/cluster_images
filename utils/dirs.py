import os
import os.path as io 
import paths 
from natsort import natsorted
from shutil import copyfile

class ApplyToFiles(object):
    def __init__(self,dir_arg=False):
        self.dir_arg=dir_arg

    def __call__(self, func):    
        @paths.path_args
        def inner_func(in_dir,out_dir):
            in_paths=get_files(in_dir,dirs=self.dir_arg)
            make_dir(out_dir)
            out_paths=[ out_dir.replace(in_i)  for in_i in in_paths]
            for in_i,out_i in zip(in_paths,out_paths):
                func(in_i,out_i)
        return inner_func

def dir_arg(func):
    def inner_func(dir_path):
        dirs=get_files(dir_path,dirs=False)
        return func(dirs)
    return inner_func

@paths.path_args
def copy_dir(in_path,out_path):
    in_files=get_files(in_path,dirs=True)
    make_dir(str(out_path))
    for in_file_i in in_files:
    	out_file_i=out_path.replace(in_file_i)
        make_dir(str(out_file_i))
        print(str(in_file_i))
        print(str(out_file_i))#out_file=out_path.
        unify_dirs(str(in_file_i),str(out_file_i))

@paths.path_args
def unify_dirs(in_path,out_path):
    dirs_paths=get_files(in_path)
    make_dir(str(out_path))
    files_paths=[]
    for dir_i in dirs_paths:
        files_paths+=get_files(dir_i,dirs=False)
    for in_file_i in files_paths:
        print(str(in_file_i))
        out_file_i=out_path.replace(in_file_i)
        print(str(out_file_i))
        copyfile(str(in_file_i),str(out_file_i))

def get_files(dir_path,dirs=True,append_path=True):
    d_path=str(dir_path)
    all_in_dir=os.listdir(d_path)
    if(dirs):    
        files= [f for f in all_in_dir  
                 if (not is_file(f,dir_path))]
    else:
    	files= [f for f in all_in_dir  
                 if is_file(f,dir_path)]
    files=natsorted(files)
    if(append_path):
        files=[paths.get_paths(dir_path,file_i) for file_i in files]
    return files

def is_file(f,path):
    file_path=str(path)+"/"+f
    return io.isfile(file_path)#io.join(path,f))

@paths.str_arg
def make_dir(path):
    if(not os.path.isdir(path)):
        os.system("mkdir "+path)

if __name__ == "__main__":
    path="../../dataset9/"
    copy_dir(path+"cats2/",path+"actions/")
    #unify_dirs(data,"test2")
    