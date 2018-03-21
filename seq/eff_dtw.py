import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np
import utils.actions
import utils.actions.read
from sets import Set

def select_cats(in_path,out_path,cats,dataset_format='cp_dataset'):
    cats=Set(cats)
    def cat_helper(action_i):
        return int(action_i.cat) in cats	
    actions=utils.actions.apply_select(in_path,out_path,img_seq=False,
    	selector=cat_helper,dataset_format=dataset_format)
#    print(len(actions))
#    save_actions=utils.actions.read.SaveActions(unorm=False,img_actions=False)
#    print(save_actions.img_actions)
#    save_actions(actions,out_path)

cats=[4, 10, 11, 15, 20]
select_cats("../../AA_dtw/skew/seq","../../AA_dtw/eff",cats)