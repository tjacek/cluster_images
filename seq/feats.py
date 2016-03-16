import utils
import utils.files as files
import basic
import seq

def to_vec_seq(conf_dir):
    action_path=conf_dir['action_path']
    actions=utils.apply_to_dir(action_path)
    extr=auto_extractor(conf_dir)
    vec_seqs=[ action_to_vecs(action_i,extr) for action_i in actions]
    return vec_seqs

def action_to_vecs(action,feat_extractor):
    print(action.cat)
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

def auto_extractor(conf_dir):
    ae_path=conf_dir["ae_path"]
    ae=files.read_object(ae_path)
    size=int(conf_dir['img_dim_x'])*int(conf_dir['img_dim_y'])
    return lambda img:apply_dec(img,ae,size)

def apply_dec(img,model,size):
    img=img.reshape((1,size)) #flatten()
    red_img=model.prediction(img)
    #print(red_img.shape)
    return red_img.flatten()

def sae_extractor():
    sae_path="../dataset7/sae"
    sae=files.read_object(sae_path)
    return lambda img:sae.features(img)  