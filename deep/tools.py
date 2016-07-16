import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np
import theano
import theano.tensor as T
import lasagne
import deep
import ae
import autoconv
import utils.imgs as imgs
import utils.dirs as dirs

def dist_to_category(dist):
    return dist.flatten().argmax(axis=0)

def to_dist(index,n_cats):
    dist=np.zeros((1,n_cats),dtype='int64')
    dist[index]=1
    return dist	

def reconstruction(ae_path,img_path,out_path):
    nn=ae.read_ae(ae_path)
    imgset=imgs.make_imgs(img_path,norm=True,conv=False)
    recon=[nn.reconstructed(img_i) 
            for img_i in imgset]
    recon=imgs.unorm(recon)
    dirs.make_dir(out_path)
    for img_i in recon:
        img_i.save(out_path)

if __name__ == "__main__": 
    ae_path="../dataset0a/ae"
    path_dir="../dataset0a/cats"
    out_path="../dataset0a/out"
    reconstruction(ae_path,path_dir,out_path)