import utils.imgs as imgs
import utils.paths
import deep.convnet
import deep.tools 
from collections import defaultdict

@utils.paths.path_args
def inspect_model(img_path,nn_path,out_path):
    imgset,model=read_assets(img_path,nn_path)
    pred_cats=predicted_cats(imgset,model)
    utils.paths.dirs.make_dir(out_path)
    for cat_i,imgset_i in pred_cats.items():
        cat_path=out_path.append('cat'+str(cat_i),copy=True)
        print(len(imgset_i))
        imgs.save_imgset(cat_path,imgset_i)

def predicted_cats(imgset,model):
    pred_cats = defaultdict(lambda:[])
    for i,img_i in enumerate(imgset):
        print(("%d " % i) + str(img_i.name))
        cat_i=model.get_category(img_i)
        pred_cats[cat_i].append(img_i)
    return pred_cats

def read_assets(img_path,nn_path):
    preproc=deep.tools.ImgPreprocProj()
    imgset=imgs.make_imgs(img_path,norm=True)
    model=deep.convnet.get_model(preproc,nn_path,compile=False,model_p=0.0)
    return imgset,model

if __name__ == "__main__":
    img_path='../inspect/full'
    nn_path='../inspect/basic_nn/nn_basic'
    out_path='../inspect/basic_cats'
    inspect_model(img_path,nn_path,out_path)