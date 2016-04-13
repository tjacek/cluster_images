import numpy as np 
import utils.actions

action_path="../dataset9/cats2/"
actions=utils.apply_to_dir(action_path)

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
    act_x=utils.action.apply_to_actions(actions,lambda act_i:act_i.org_dim[0])
    act_y=utils.actions.apply_to_actions(actions,lambda act_i:act_i.org_dim[1])
    act_size=utils.actions.apply_to_actions(actions,lambda act_i:np.product(act_i.shape))
    print("dim x")
    show_stats(act_x)
    print("dim y")
    show_stats(act_y)
    print("x*y")
    show_stats(act_size)

def extract_data(actions,fun=len):
    return [fun(action_i) for action_i in actions]



action_stats(actions)