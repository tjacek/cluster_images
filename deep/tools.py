import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np
import theano
import theano.tensor as T
import lasagne
import deep
import ae
#import autoconv
import utils.imgs as imgs
import utils.paths.dirs as dirs
import deep.reader

class ImgPreproc(object):
    def __init__(self,dim):
        self.dim=dim

    def __call__(self,imgset):
        x,y=get_dims(imgset)
        new_height=self.get_new_height(x)
        def reshape_helper(img_i):
            frames=np.vsplit(img_i,self.dim) 
            splited_img=np.stack(frames)
            print(np.max(splited_img))
            return splited_img
        return [ reshape_helper(img_i) for img_i in imgset]

    def get_new_height(self,x):
        if(x % self.dim != 0):
            raise Exception("incorrect img size")
        return x/self.dim

def get_dims(imgset):
    first=imgset[0]
    return first.shape[0],first.shape[1] 

#def show_imgs(imgs3D):
#    for img_i in imgs3D:
#        print(type(img_i))
#        print(img_i.shape)

#def dist_to_category(dist):
#    return dist.flatten().argmax(axis=0)

#def to_dist(index,n_cats):
#    dist=np.zeros((1,n_cats),dtype='int64')
#    dist[index]=1
#    return dist	

#def preproc2D(in_img):
#    org_img=in_img.get_orginal()
#    img3D=np.expand_dims(org_img,0)
#    img4D=np.expand_dims(img3D,0)
#    return img4D

#def preproc3D(in_img):
#    org_img=in_img.get_orginal()
#    img3D=imgs.split_img(org_img)
#    img4D=np.expand_dims(img3D,0)
#    return img4D

#def preprocPost(in_img): 
#    org_img=in_img.get_orginal()
#    img3D=imgs.split_img(org_img,scale=3)
#    img4D=np.expand_dims(img3D,0)
#    return img4D

#def postproc3D(in_img):
#    img3D=np.squeeze(in_img)
#    img2D=imgs.unify_img(img3D)
#    return img2D

#def reconstruction(ae_path,img_path,out_path):
#    reader=deep.reader.NNReader()
#    nn=reader.read(ae_path)
#    imgset=imgs.make_imgs(img_path,norm=True)
#    recon=[nn.reconstructed(img_i) 
#            for img_i in imgset]
#    recon=imgs.unorm(recon)
#    dirs.make_dir(out_path)
#    for i,img_i in enumerate(recon):
#        img_i.save(out_path,i)

#def check_transform(img_path,out_path):
#    imgset=imgs.make_imgs(img_path,norm=False,transform=imgs.to_3D)
#    dirs.make_dir(out_path)
#    for i,img_i in enumerate(imgset):
#        raw_i=img_i[0]#imgs.unify_img(img_i)
#        print(raw_i.shape)
#        name_i="img"+str(i)+".jpg"
#        new_img_i=imgs.Image(name_i,raw_i)
#        new_img_i.save(out_path)

if __name__ == "__main__": 
    ae_path="../dataset1/conv_ae_"
    img_path="../dataset1/cats"
    out_path="../dataset1/recon"
    reconstruction(ae_path,img_path,out_path)
