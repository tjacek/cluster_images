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
import deep.reader

class ImgPreproc(object):

    def apply(self,in_img):
        org_img=in_img.get_orginal()
        img3D=imgs.split_img(org_img,3)
        img4D=np.expand_dims(img3D,0)
        return img4D

    def __call__(self,imgset):
        raw_imgs=self.basic(imgset)
        imgs3D=[ imgs.split_img(img_i,scale=3)
                 for img_i in raw_imgs]
        vol=np.array(imgs3D)
        return vol

    def basic(self,imgset):
        return [img_i.get_orginal()
                for img_i in imgset]

def dist_to_category(dist):
    return dist.flatten().argmax(axis=0)

def to_dist(index,n_cats):
    dist=np.zeros((1,n_cats),dtype='int64')
    dist[index]=1
    return dist	

def preproc2D(in_img):
    org_img=in_img.get_orginal()
    img3D=np.expand_dims(org_img,0)
    img4D=np.expand_dims(img3D,0)
    return img4D

def preproc3D(in_img):
    org_img=in_img.get_orginal()
    img3D=imgs.split_img(org_img)
    img4D=np.expand_dims(img3D,0)
    return img4D

def postproc3D(in_img):
    img3D=np.squeeze(in_img)
    img2D=imgs.unify_img(img3D)
    return img2D

def reconstruction(ae_path,img_path,out_path):
    reader=deep.reader.NNReader()
    nn=reader.read(ae_path)#autoconv.read_conv_ae(ae_path)
    imgset=imgs.make_imgs(img_path,norm=True)
    recon=[nn.reconstructed(img_i) 
            for img_i in imgset]
    recon=imgs.unorm(recon)
    dirs.make_dir(out_path)
    for i,img_i in enumerate(recon):
        img_i.save(out_path,i)

def check_transform(img_path,out_path):
    imgset=imgs.make_imgs(img_path,norm=False,transform=imgs.to_3D)
    dirs.make_dir(out_path)
    for i,img_i in enumerate(imgset):
        raw_i=img_i[0]#imgs.unify_img(img_i)
        print(raw_i.shape)
        name_i="img"+str(i)+".jpg"
        new_img_i=imgs.Image(name_i,raw_i)
        new_img_i.save(out_path)

if __name__ == "__main__": 
    ae_path="../dataset1/conv_ae_"
    img_path="../dataset1/cats"
    out_path="../dataset1/recon"
    #check_transform(img_path,out_path)
    reconstruction(ae_path,img_path,out_path)