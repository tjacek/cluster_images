import utils
import utils.files as files
import basic
import seq
import numpy as np

def to_vec_seq(conf_dir):
    action_path=conf_dir['action_path']
    actions=utils.apply_to_dir(action_path)
    extr_builder=EXTRACTORS[conf_dir['features']]
    extr=extr_builder(conf_dir)
    #extr=auto_extractor(conf_dir)
    vec_seqs=[ action_to_vecs(action_i,extr) for action_i in actions]
    return vec_seqs

def action_to_vecs(action,feat_extractor):
    print(action.cat)
    vecs=[feat_extractor(frame_i) for frame_i in action.frames]
    return seq.Seq(vecs,action.cat)

def basic_feat(img):
	img=img.reshape((60,60))
	return basic.get_features(img)

def int_cats(cats):
    cat_to_int={}	
    #for action_i in actions:
    for cat_i in cats:
        cat_id=cat_to_int.get(cat_i,None)
        if(cat_id==None):
            cat_to_int[cat_i]=len(cat_to_int.keys())
    return cat_to_int

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

def sae_extractor(conf_dir):
    sae_path=conf_dir['sae_path']
    sae=files.read_object(sae_path)
    return lambda img:sae.features(img)  

def cat_extractor(conf_dir,vector=True):
    print(conf_dir.keys())
    sae_path=conf_dir['sae_path']
    sae=files.read_object(sae_path)
    if(vector):
        def clos_extractor(img):
            cat_i=sae.get_category(img)
            return get_vector_cat(cat_i)
    #    return clos_extractor
    else:
         def clos_extractor(img):
            return sae.get_category(img)
    return clos_extractor

EXTRACTORS={'auto':auto_extractor ,'sae':sae_extractor,'cat':cat_extractor}

def get_vector_cat(cat_i):
    vec=np.zeros((12,))
    vec[cat_i]=1
    return vec
