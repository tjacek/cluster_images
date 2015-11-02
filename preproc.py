import data
import deep.autoencoder as ae

def create_autoencoder(image_path,out_path):
    images=data.read_images(image_path)
    da=ae.learning_autoencoder(images) 
    ae.reconstruct_images(images,da,out_path)

if __name__ == "__main__":
    path="/home/user/cls/"
    in_path=path+"test/"
    out_path=path+"out/"
    create_autoencoder(in_path,out_path)
