import data,deep,utils
import deep.autoencoder as ae
import theano

def create_autoencoder(image_path,obj_path):
    images=data.read_images(image_path)
    da=ae.learning_autoencoder(images)
    utils.save_object(da.model,obj_path) 
    print("autoencoder saved as " + obj_path)

def create_images(image_path,out_path):
    images=data.read_images(image_path)
    da=ae.learning_autoencoder(images) 
    ae.reconstruct_images(images,da,out_path)

def read_autoencoder(obj_path):
    model=utils.read_object(obj_path)
    return ae.AutoEncoder(model)

def deep_reduce(dataset):
    path="/home/user/cls/dp/ae"
    da=read_autoencoder(path)
    print(dataset)
    for inst in dataset.instances:
        reduced=da.test(inst.data)
        inst.data=reduced.flatten()

if __name__ == "__main__":
    path="/home/user/cls/"
    in_path=path+"test/"
    out_path=path+"out/"
    obj_path=path+"dp/ae"
    create_autoencoder(in_path,obj_path)
