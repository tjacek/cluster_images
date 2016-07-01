import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import utils.paths as paths
import utils.dirs as dirs
import utils.imgs as imgs
import utils.files as files

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
    return img_i[10],img_i[20]

if __name__ == "__main__":
    path='../dataset1/cats/'
    #print([ str(path_i) for path_i in dirs.bottom_dirs(path)])
    extract_features(path)
    #extract_features(sys.argv[1])