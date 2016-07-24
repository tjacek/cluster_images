import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np
import utils.paths as paths
import utils.dirs as dirs
import utils.imgs as imgs
import utils.files as files
import utils.pcloud as pcloud
import basic
import basic.external as ext
import deep.reader 

@paths.path_args
def extract_features(in_path,nn_path):
    out_path=in_path.copy().set_name('seq')
    print(str(out_path))
    nn_reader=deep.reader.NNReader()
    nn_extractor=nn_reader.read(nn_path)
    transform_seq(in_path,out_path,nn_extractor)

@dirs.apply_to_dirs
def transform_seq(in_path,out_path,extractor):
    imgs_seq= imgs.make_imgs(in_path,norm=True) #imgs.read_images(in_path)
    seq=[extractor(img_i) 
                for img_i in imgs_seq]
    txt=files.seq_to_string(seq)
    print(str(in_path))
    print(str(out_path))
    files.save_string(str(out_path)+'.txt',txt)

if __name__ == "__main__":
    path='../dataset1/cats/'
    extract_features(path)