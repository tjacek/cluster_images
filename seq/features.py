import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np
import utils.paths as paths
import utils.paths.dirs as dirs
import utils.imgs as imgs
import utils.paths.files as files
import utils.pcloud as pcloud
import basic
import basic.external as ext
        
#def transform_external(in_file,out_file,short_names,dataset_format='cp_dataset'):
#    extractor=ext.read_external(str(in_path),short_names) 
#    action_reader=utils.actions.read.ReadActions(dataset_format)

@paths.path_args
def extract_features(in_path,ext_path,out_path,
                                      short_names=True):
    #out_path=in_path.copy().set_name(str(seq_path))
    dirs.make_dir(str(out_path))
    extractor=ext.read_external(str(ext_path),short_names)
    transform_seq(in_path,out_path,extractor)

@dirs.apply_to_dirs
def transform_seq(in_path,out_path,extractor):
    imgs_seq= imgs.make_imgs(in_path,norm=True) #imgs.read_images(in_path)
    print(str(extractor))
    seq=[extractor(img_i)
                for img_i in imgs_seq]
    seq=[ seq_i.flatten()  
           for seq_i in seq
             if seq_i!=None]
    txt=files.seq_to_string(seq)
    print(str(in_path))
    print(str(out_path))
    files.save_string(str(out_path)+'.txt',txt)

if __name__ == "__main__":
    path='../dataset2a/full/'
    ext_path='../dataset2a/conv_nn.txt'
    extract_features(path,ext_path,'seq')