import data,deep,utils
import deep.autoencoder as ae
import theano
#import theano.misc.pkl_utils as qwerty

def create_autoencoder(image_path,obj_path):
    images=data.read_images(image_path)
    da=ae.learning_autoencoder(images)
    #qwerty.dump(da,obj_path) 
    utils.save_object(da.model,obj_path) 
    print("autoencoder saved as " + obj_path)

def create_images(image_path,out_path):
    images=data.read_images(image_path)
    da=ae.learning_autoencoder(images) 
    ae.reconstruct_images(images,da,out_path)

if __name__ == "__main__":
    path="/home/user/cls/"
    in_path=path+"test/"
    out_path=path+"out/"
    obj_path=path+"dp/ae"
    create_autoencoder(in_path,obj_path)
