import re

class Path(object):
    def __init__(self, text):
        if(type(text)==Path):
            text=str(text)
        else:
            text=re.sub(r'(//)+','/',text)
        self.items=[]
        self.add(str(text))

    def __getitem__(self,i):
        return self.items[i]

    def __len__(self):
        return len(self.items)

    def __add__(self, other):
        copy=self.copy()
        if(type(other)==list):
            copy.append(other)
        if(type(other)==Path):
            copy.append(other.items)
        return copy

    def __str__(self):
        s="/".join(self.items)	
        s=re.sub(r'(//)+','/',s)
        return s

    def exchange(self,old,new):
        str_path=str(self)
        str_path=str_path.replace(old,new)
        return Path(str_path)

    def replace(self,other_path):
        new_path=self.copy()
        name=other_path.get_name()
        new_path.append(name)
        return new_path

    def get_name(self):
        return self.items[-1]

    def set_name(self,name):
        self.items[-1]=name
        return self

    def add(self,str_path):
        strs=str_path.split("/")
        for str_i in strs:
            if(str_i!=''):
                self.append(str_i)
    
    def append(self,items,copy=False):
        if(copy):
            path_i=self.copy()
        else:
            path_i=self
        if(type(items)==str):
            items=[items]
        for item_i in items:
    	    item=item_i.replace("/","")
            path_i.items.append(item)
        return path_i

    def copy(self):	
        str_path=str(self)
        return Path(str_path)

    def first(self,k):
        new_path=self.copy()
        new_path.items=self.items[0:len(self)-k]
        return new_path

def get_paths(path,filename):
    if(type(path)==str):
        path=Path(path)
    path=path.copy()
    path.add(filename)
    return path

def path_args(func):
    def path_fun(*args): 
        path_args=[ str_to_path(arg_i) 
                      for arg_i in args]
        return func(*path_args)
    return path_fun        

def str_arg(func):
    def inner_fun(*args):
        in_strs=[str(arg_i) for arg_i in args]
        return func(*in_strs)
    return inner_fun

def str_to_path(obj):
    if(type(obj)==str):
        return Path(obj)
    return obj

def to_paths(files):
    return [Path(file_i) for file_i in files]

if __name__ == "__main__":
    data="../dataset2/binary"
    path=Path(data)
    path.add("/abc/aff")
    print(str(path))