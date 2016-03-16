import utils
import utils.files as files
import basic
import seq

def to_vec_seq(action_path):
    actions=utils.apply_to_dir(action_path)
    extr=auto_extractor()
    vec_seqs=[ action_to_vecs(action_i,extr) for action_i in actions]
    return vec_seqs

def action_to_vecs(action,feat_extractor):
    vecs=[feat_extractor(frame_i) for frame_i in action.frames]
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

def auto_extractor():
    ae_path="../dataset6/ae"
    ae=files.read_object(ae_path)
    return lambda img:ae.apply(img)  