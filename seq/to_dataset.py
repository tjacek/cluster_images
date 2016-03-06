import utils
import utils.files as files
import seq
import numpy as np 

def to_dataset(in_path,out_path):
    str_seq=files.read_file(in_path)
    cat_names={}
    instances=[ seq_to_instances(str_seq_i,cat_names) 
                    for str_seq_i in str_seq]
    files.save_array(instances,out_path)    

def seq_to_instances(str_seq,cat_names):
    raw_action=str_seq.split("#") 
    hist=get_histogram(raw_action[0])
    cat_index=utils.get_value(raw_action[1],cat_names)
    instance=files.vector_string(hist)+"#"+str(cat_index)	
    return instance

def get_histogram(raw_seq,n_cats=7):
    hist=np.zeros((n_cats,))
    for frame_cat in raw_seq:
    	index=seq.get_cat(frame_cat)
    	hist[index]+=1.0
    hist/=np.amax(hist)
    return hist    
