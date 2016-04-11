import utils.actions as action
import utils.files as files
import utils.dirs #as files
import os

def apply_to_dir(in_path):
    all_dirs=find_all_dirs(in_path)
    action_dirs=[dir_i for dir_i in all_dirs
                   if is_action(dir_i)]
    actions=[action.read_action(dir_i)  for dir_i in action_dirs]
    #for dir_i in action_dirs:
    #    print(str(dir_i))
    return actions

def get_value(cat,mydict):
    if(not (cat in mydict)):
        mydict[cat]=len(mydict.keys())
    return mydict[cat]

def is_action(dir_path):
    dir_paths=utils.dirs.get_files(dir_path,dirs=True)
    files_paths=utils.dirs.get_files(dir_path,dirs=False)
    return len(dir_paths)==0 and len(files_paths)!=0

def find_all_dirs(dir_path):
    dirs=utils.dirs.get_files(dir_path)
    all_dirs=[]
    for dir_i in dirs:
        all_dirs+=find_all_dirs(dir_i)
    all_dirs+=dirs
    return all_dirs