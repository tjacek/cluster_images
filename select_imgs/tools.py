import utils.files
import utils.dirs
import utils.paths

def save_clustering(out_path, imgs,img_cls):
    lines=[utils.files.vector_string(img_i) + ",#" + str(cls_i) 
                             for img_i,cls_i in zip(imgs,img_cls)]
    #lines=[img_i +"#" + str(cls_i) for img_i,cls_i in dataset]
    lines=utils.files.array_to_txt(lines,"\n")
    utils.files.save_string(out_path,lines) 

def create_dir_struct(config,n_clusters):
    out_path=utils.paths.Path(config["out_path"])
    utils.dirs.make_dir(out_path)
    out_path.append(config["cls_alg"])
    utils.dirs.make_dir(out_path)
    n_clusters+=1
    cls_dirs=[out_path + ["cls"+str(i)] for i in range(n_clusters)]
    for c_dir in cls_dirs:
        utils.dirs.make_dir(c_dir)
    return out_path

def split_clusters(config,dataset,n_clusters):
    out_path=create_dir_struct(config,n_clusters)
    i=0
    for img_i,cls_i in dataset:
        cls_dir="cls"+str(cls_i)
        if(cls_i>-1):
            txt_id="fr"+str(i)+".jpg"
            i+=1
            full_path=out_path
            full_path=out_path+[cls_dir,txt_id]
            print(str(full_path))
            utils.imgs.save_img(full_path,img_i)