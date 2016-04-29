import deep
import utils.imgs ,utils.dirs,utils.files
import deep.autoencoder as ae
import utils.conf
from utils.dirs import dir_arg, ApplyToFiles
import cv2
import numpy as np

def create_autoencoder(conf_dict):
    image_path=conf_dict["img_path"]
    obj_path=conf_dict["ae_path"]
    print("read images from "+ image_path)
    dim_x=int(conf_dict['dim_x'])
    dim_y=int(conf_dict['dim_y'])
    #utils.dirs.make_dir(obj_path)
    #@ApplyToFiles(True)
    def inner_func(in_path,out_path):
        print(str(in_path))
        print(str(out_path))
        imgs=utils.imgs.read_images(str(in_path))
        print("Number of images %i",len(imgs))
        da=deep.train_model_unsuper(imgs,ae.default_parametrs(),num_iter=500,input_dim=(dim_x,dim_y))
        utils.files.save_object(da,str(out_path)) 
        print("autoencoder saved as " + str(out_path))
    inner_func(image_path,obj_path)

def create_images(image_path,out_path):
    images=data.read_images(image_path)
    da=ae.learning_autoencoder(images) 
    ae.reconstruct_images(images,da,out_path)

def read_autoencoder(obj_path):
    model=utils.read_object(obj_path)
    return ae.AutoEncoder(model)

@ApplyToFiles(dir_arg=True)
def apply_ae(in_path,out_path,ae_path):
    ae_model=utils.files.read_object(ae_path)
    imgs=utils.imgs.read_images(in_path)
    transform_img=lambda img_i:ae_model.apply(img_i)
    transformed_imgs=[transform_img(img_i) for img_i in imgs]
    utils.files.make_dir(out_path)
    for i,trans_img_i in enumerate(transformed_imgs):
        full_path=out_path+"/"+"img_" + str(i) +".jpg"
        trans_img_i=np.reshape(trans_img_i,(120,60))
        trans_img_i*=250.0
        cv2.imwrite(full_path,trans_img_i)
   

if __name__ == "__main__":
    conf_path="conf/dataset1.cfg"
    conf_dict=utils.conf.read_config(conf_path)
    #create_autoencoder(conf_dict)
    apply_ae(conf_dict["action_path"],"recon1",conf_dict["ae_path"])
