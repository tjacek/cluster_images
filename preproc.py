import deep
import utils.imgs ,utils.dirs#as image
import deep.autoencoder as ae
import utils.conf
from utils.dirs import dir_arg, ApplyToFiles

def create_autoencoder(conf_dict):
    image_path=conf_dict["img_path"]
    obj_path=conf_dict["ae_path"]
    print("read images from "+ image_path)
    dim_x=int(conf_dict['dim_x'])
    dim_y=int(conf_dict['dim_y'])
    utils.dirs.make_dir(obj_path)
    @ApplyToFiles(True)
    def inner_func(in_path,out_path):
        print(str(in_path))
        print(str(out_path))
        imgs=utils.imgs.read_images(str(in_path))
        print("Number of images %i",len(imgs))
        da=deep.train_model_unsuper(imgs,ae.default_parametrs(),input_dim=(dim_x,dim_y))
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

def apply_ae(image_path,ae_path):
    imgs=utils.imgs.read_img_dir(image_path)
    #ae.read_autoencoder(ae_path)
    reduced=ae.apply_autoencoder(imgs,ae_path)
    lines=[utils.files.vector_string(img_i) for img_i in reduced]
    lines=utils.files.array_to_txt(lines,"\n")
    utils.files.save_string("auto.csv",lines)

if __name__ == "__main__":
    conf_path="conf/dataset9.cfg"
    conf_dict=utils.conf.read_config(conf_path)
    create_autoencoder(conf_dict)
    #apply_ae(img_path,ae_path)
