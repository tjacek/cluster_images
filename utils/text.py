import re
import paths.files 

class ExtractNumber(object):
    def __init__(self):
        self.pattern=re.compile(r"\d+")

    def __call__(self,name):
        if(type(name)==paths.Path):
            name=name.get_name()
        raw_digits=re.findall(self.pattern,str(name))
        return int(raw_digits[0]) 

def get_person(name):
    raw=name.split('_')[0]
    return extract_number(raw)
    
def get_category(name):
    raw=name.split('_')[0]
    return extract_number(raw)

def extract_sufix(filename):
    return filename.split(".")[-1] 

def replace_sufix(sufix,files):
    return [s.replace(sufix,"") for s in files]

def has_sufix(filename,sufix):
	return extract_sufix(filename)==sufix