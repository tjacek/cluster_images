import utils.actions as action
import utils.files as files

def apply_to_dir(in_path): 
    all_dirs=files.get_dirs(in_path,True)
    action_i=action.read_action(in_path)
    #print(in_path)
    if(len(action_i)==0):
        actions=[]
    else:
    	actions=[action_i]
    dir_actions=[apply_to_dir(dir_i) for dir_i in all_dirs]
    for dir_actions_i in dir_actions:
        actions+=dir_actions_i
    return actions