import deep
import utils.imgs #as image
import deep.autoencoder as ae
#import theano

def create_autoencoder(image_path,obj_path):
    imgs=utils.imgs.read_img_as_array(image_path)
    da=ae.train_model(imgs,ae.default_parametrs())
    utils.files.save_object(da.get_model(),obj_path) 
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
    img_path="test"   #"../dataset6/imgs"
    ae_path="../dataset/ae" #path+"dp/ae"
    create_autoencoder(img_path,ae_path)
    #apply_ae(img_path,ae_path)
