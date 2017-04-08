import paths.files
import shutil
import utils.actions

class SelectModulo(object):
    def __init__(self,m=0):
        self.m=m

    def __call__(self,action):
        if(type(action)==utils.actions.Action):
            n=action.person
        return (n % 2)==self.m

class SelectPerson(object):
    def __init__(self, n_person,neg):
        self.n=n_person
        self.neg=neg
        
    def __call__(self,action):
        bool_value=(self.n==action.person)
        if(self.neg):
            bool_value=not bool_value
        return bool_value    
    
if __name__ == "__main__":
    path="../dataset/"
    in_path=path+"MSRDailyAct/"
    out_path=path+"binary"
    select_data(in_path,out_path)