import io
import deep.autoencoder as ae

def create_autoencoder(image_path):
    images=io.read_images(image_path)
    da=ae.learning_autoencoder(images) 
    print(len(images))

if __name__ == "__main__":
    path="/home/user/cls/"
    in_path=path+"actions/"
    create_autoencoder(in_path)
