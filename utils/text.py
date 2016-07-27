import re
import files 

def get_person(name):
    raw=name.split('_')[0]
    return extract_number(raw)
    
def get_category(name):
    raw=name.split('_')[0]
    return extract_number(raw)
    
def extract_number(name):
    pattern = re.compile(r"\d+")
    raw_digits=re.findall(pattern,str(name))[0]
    return int(raw_digits)

def extract_sufix(filename):
    return filename.split(".")[-1] 

def replace_sufix(sufix,files):
    return [s.replace(sufix,"") for s in files]

def has_sufix(filename,sufix):
	return extract_sufix(filename)==sufix