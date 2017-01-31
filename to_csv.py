import re

def get_csv(line):
    if(line.isalnum()):
        return ",".join(line.split())
    else:
        return None	