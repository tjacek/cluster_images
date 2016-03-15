import utils
import utils.imgs
import deep.autoencoder as ae
import shape_context
import basic

def use_autoencoder(in_path):
    imgs=utils.imgs.read_img_dir(in_path)
    print("read data")
    reduced_img=ae.apply_autoencoder(imgs,config['ae_path'])
    return reduced_img

def use_shape_context(in_path):
    #imgs=utils.imgs.read_img_dir(in_path)
    img_paths=utils.files.get_files(in_path)
    red_imgs=[shape_context.get_shape_context(img_path_i)
                  for img_path_i in img_paths]
    red_imgs=[img_i for img_i in red_imgs
                      if img_i!=None]
    return red_imgs#,imgs

def use_basic(in_path):
    imgs=utils.imgs.read_img_dir(in_path,False)
    print("read data")
    basic_img=[basic.get_features(img_i) for img_i in imgs]
    return basic_img 
