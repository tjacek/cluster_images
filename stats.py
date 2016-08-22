import numpy as np 
import utils.actions

def action_stats(actions):
    actions_len=extract_data(actions)
    show_stats(actions_len)
    dim_stats(actions)

def show_stats(actions_len):
    avg_act=np.average(actions_len)#float(sum(actions_len))/float(len(actions_len))
    max_act=max(actions_len)
    min_act=min(actions_len)
    med_act=np.median(actions_len)
    print("Avg "+ str(avg_act))
    print("Max " + str(max_act))
    print("Min " + str(min_act))
    print("Mediana " + str(med_act))

def dim_stats(actions):
    fun_x=lambda(img_i): img_i.org_dim[0]
    act_x=utils.actions.apply_to_imgs(fun_x,actions)
    act_x=utils.unify_list(act_x)
    fun_y=lambda(img_i): img_i.org_dim[1]
    act_y=utils.actions.apply_to_imgs(fun_y,actions)   
    act_y=utils.unify_list(act_y)
    print("dim x")
    show_stats(act_x)
    print("dim y")
    show_stats(act_y)
    act_size=[ x_i*y_i for x_i,y_i in zip(act_x,act_y)]
    print("x*y")
    show_stats(act_size)

def extract_data(actions,fun=len):
    return [fun(action_i) for action_i in actions]

if __name__ == "__main__": 
    action_path="../dataset7/cats/"
    actions=utils.actions.read_actions(action_path)
    action_stats(actions)