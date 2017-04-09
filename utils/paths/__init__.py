import re

class Path(object):
    def __init__(self, text):
        if(type(text)==list):
            text='/'.join(text)
        if(type(text)==Path):
            text=str(text)
        assert(type(text)==str)
        self.items=[]
        for text_i in text.split('/'):
            self << text_i

    def __getitem__(self,i):
        return self.items[i]

    def __len__(self):
        return len(self.items)

    def __lshift__(self,item_i):
        item_i=clean_item(item_i)
        self.items.append(item_i)

    def __add__(self,other):
        return self.append(other,copy=True)

    def __str__(self):
        return "/".join(self.items)  

    def get_name(self):
        return self.items[-1]
    
    def set_name(self,name,copy=False):
        if(copy):
            new_path=self.copy()
        else:
            new_path=self    
        new_path.items[-1]=clean_item(name)
        return new_path
    
    def copy(self): 
        str_path=str(self)
        return Path(str_path)

    def append(self,new_items,copy=False):
        if(type(new_items)==Path):
            new_items=new_items.items#str(new_items)
        if(type(new_items)==str):
            new_items=new_items.split("/")
        assert(type(new_items)==list)
        extended_path=self.get_path(copy)
        for str_i in new_items:
            if(str_i!=''):
                extended_path << str_i
        return extended_path

    def get_path(self,copy=False):
        if(copy):
            return self.copy()
        else:
            return self

    def exchange(self,old,new):
        str_path=str(self)
        str_path=str_path.replace(old,new) #We use string method
        return Path(str_path)

    def replace(self,other_path):
        new_path=self.copy()
        name=other_path.get_name()
        new_path << (name)
        return new_path

    def get_postfix(self):
        name=self.get_name()
        name_split=name.split(".")
        if(len(name_split) < 2):
            raise Exception("no postfix in " + name)
        return name_split[1]    

    def first(self,k):
        new_path=self.copy()
        new_path.items=self.items[0:len(self)-k]
        return new_path

    def subpaths(self):
        sub_paths=[]
        current_path=[]
        for item_i in self.items:
            current_path.append(item_i)
            sub_paths.append(Path(current_path))
        return sub_paths

def clean_item(item_i):
    if(type(item_i)==Path):
        item_i=item_i.get_name()
    item_i=re.sub(r'(//)+','/',item_i)
    return item_i

def get_paths(path,filename):
    if(type(path)==str):
        path=Path(path)
    path=path.copy()
    path.append(filename)
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