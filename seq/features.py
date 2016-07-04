import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import utils.paths as paths
import utils.dirs as dirs
import utils.imgs as imgs
import utils.files as files
import utils.pcloud as pcloud
import basic

@paths.path_args
def extract_features(in_path):
    out_path=in_path.copy().set_name('seq')
    print(str(out_path)) 
    transform_seq(in_path,out_path)

@dirs.apply_to_dirs
def transform_seq(in_path,out_path):
    imgs_seq=imgs.read_images(in_path)
    seq=[extr_feat(img_i) 
                for img_i in imgs_seq]
    txt=files.seq_to_string(seq)
    print(str(in_path))
    print(str(out_path))
    files.save_string(str(out_path)+'.txt',txt)

def extr_feat(img_i):
    #org_img=img_i.get_orginal()
    cloud_i=pcloud.make_point_cloud(img_i)
    cloud_i=pcloud.normalized_cloud(cloud_i)
    return cloud_i.center_of_mass()
    #basic.get_features(half_img)

if __name__ == "__main__":
    path='../dataset0/cats/'
    extract_features(path)