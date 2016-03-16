import utils
import basic
import seq

def to_vec_seq(action_path):
    actions=utils.apply_to_dir(action_path)
    vec_seqs=[ action_to_vecs(action_i) for action_i in actions]
    return vec_seqs

def action_to_vecs(action):
    vecs=[basic_feat(frame_i) for frame_i in action.frames]
    return seq.Seq(vecs,action.cat)

def basic_feat(img):
	img=img.reshape((60,60))
	return basic.get_features(img)

def int_cats(action_i):
    cat_to_int={}	
	for action_i in actions:
        cat_id=cat_to_int.get(action_i.cat,None)
        if(cat_id==None):
            cat_to_int[cat_id]=len(cat_to_int.keys())	 