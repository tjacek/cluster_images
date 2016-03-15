import deep
import utils.imgs #as image
import deep.autoencoder as ae
import utils.conf

def create_autoencoder(conf_dict):
    image_path=conf_dict["img_path"]
    obj_path=conf_dict["ae_path"]
    print("read images from "+ image_path)
    imgs=utils.imgs.read_img_dir(image_path)
    da=deep.train_model_unsuper(imgs,ae.default_parametrs())
    utils.files.save_object(da,obj_path) 
    print("autoencoder saved as " + obj_path)

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
    conf_path="conf/dataset6.cfg"
    conf_dict=utils.conf.read_config(conf_path)
    create_autoencoder(conf_dict)
    #apply_ae(img_path,ae_path)
