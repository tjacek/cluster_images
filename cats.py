import deep
import utils.imgs
import utils.data as data
import deep.sae as sae
import utils.conf
import utils.files as files

def create_sae(dir_path,conf_path,out_path):
    X,y=data.read_dataset(dir_path)
    conf_dict=utils.conf.read_config(conf_path)
    model=get_sae(conf_dict)
    deep.train_model_super(X,y,model) 
    files.save_object(model,out_path)

def get_sae(conf_dict):
    #image_path=conf_dict["img_path"]
    ae_path=conf_dict["ae_path"] 
    ae=files.read_object(ae_path) 	
    return sae.StackedAE(ae,7)
    
if __name__ == "__main__":
    path="/home/user/reps/dataset6/"
    final_path=path+"cats"
    dir_path="/home/user/reps/cluster_images/dataset"
    conf_path="conf/dataset6.cfg"
    out_path=path+"sae"
    create_sae(dir_path,conf_path,out_path)