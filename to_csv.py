import re
import utils.dirs

class GetCSV(object):
    def __init__(self):
        self.regex =re.compile('([0-9]|(\[)|(\])|(\.)|(\s))+')
        self.braces= re.compile('(\[)|(\])')
        self.postfix= re.compile('(\.)(.)+')

    def to_dir(self,dir_path):
        all_paths=utils.dirs.all_files(dir_path)
        for path_i in all_paths:
            if(path_i.get_postfix()=='txt'):
                self(str(path_i))

    def __call__(self,filename):
        csv=''
        with open(filename) as f:
            lines = f.readlines()
        for line_i in lines:    
            csv+=self.get_csv(line_i)
        csv_name=re.sub(self.postfix,'.csv',filename)
        output_file = open(csv_name, 'w')
        output_file.write(csv)

    def get_csv(self,line):
    	print(line)
        if(self.check_line(line)):
            line=re.sub(self.braces,'',line)
            return ",".join(line.split()) +'\n'
        else:
            return None	

    def check_line(self,line):
        #return re.search(self.regex,line)!=None
        return self.regex.search(line)!=None

if __name__ == "__main__":
    path='Documents/artykul/podsumowanie/' 
    #nast_selekcja_dtw.txt'
    get_csv=GetCSV()
    get_csv.to_dir(path)