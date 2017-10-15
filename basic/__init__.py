import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import cv2
import numpy as np
import utils.pcloud #as pcloud
import scipy.stats
import scipy.stats.stats as st
from sklearn.decomposition import PCA
import utils
import utils.imgs
import utils.actions

class ExtractFeatures(object):
    def __init__(self,preproc,features):
        self.preproc=preproc
        self.features=features

    def __call__(self,imgs_seq):
        img_seq=[ self.preproc(img_i) for img_i in imgs_seq ]
        return [ self.features(img_i)
                 for img_i in img_seq]

class SimplePreproc(object):
    def __init__(self, div=3):        
        self.div=div

    def __call__(self,img_i):
        new_size=img_i.shape[0]/self.div
        return img_i[:][0:new_size]

def action_features(in_path,out_path,extractor,dataset_format='cp_dataset'):
    read_actions=utils.actions.read.ReadActions(dataset_format,norm=True)
    actions=read_actions(in_path)
    new_actions=[ action_i(extractor) for action_i in actions]
    save_actions=utils.actions.read.SaveActions(img_actions=False)
    save_actions(new_actions,out_path)

def test_transform(img_i):
    print(img_i.shape)
    return [img_i.shape[0],img_i.shape[1]]

def make_extract():
    return ExtractFeatures(SimplePreproc(),PcloudFeatures())

class GetFeatures(object):
    def __init__(self, cloud_extractors):
        self.cloud_extractors=cloud_extractors
        
    def __call__(self,img):
        img=preproc_img(img)
        print('Shape:')
        print(img.shape)
        points=utils.pcloud.make_point_cloud(img)
        #points=pcloud.normalized_cloud(points)
        if(points==None or len(points)==0):
    	    #return None
            raise Exception("No points")
        points=utils.pcloud.unit_normalized(points)
        all_feats=[]
        for extr_i in self.cloud_extractors:
            all_feats+=extr_i(img,points)
        print(all_feats)
        print(len(all_feats))      	
        return np.array(all_feats)

def get_basic_features():
    #cloud_extractors=[area_feat,skewness_features,std_features,corl_features]
    cloud_extractors=[extr_point]
    #cloud_extractors=[corl_features]
    return GetFeatures(cloud_extractors)

def preproc_img(img_i):
    x=img_i.shape[0]
    y=img_i.shape[1]
    #z= x/(x/y)   
    return img_i[0:y]

def extr_features(img,pcloud):
    extr=pcloud.max(0,1) 
    extr+=pcloud.min(0,1)
    extr+=pcloud.max(1,0) 
    extr+=pcloud.min(1,0)
    print('extr')
    print(extr)
    return extr

def center(img,pcloud):
    return pcloud.center_of_mass(True)

def extr_point(img_i,pcloud):
    index=np.unravel_index(img_i.argmax(),img_i.shape)
    return list(index)

def std_features(img,pcloud):
    points=pcloud.get_numpy()
    feat=np.std(points,axis=0)
    feat[0]/=float(img.shape[0])
    feat[1]/=float(img.shape[0])
    #feat[2]/=250.0
    return list(feat)

def skewness_features(img,pcloud):
    feats=[]
    points=pcloud.get_numpy()
    dim=pcloud.dims
    for x_i in range(dim):    
        print(points[:,x_i].shape)
        corr_xy=st.skew(points[:,x_i])
        feats.append(corr_xy)    
    #print(feats)
    return feats

def pca_features(img,pcloud):
    feats=[]
    pca = PCA(n_components=3)
    pca.fit(pcloud.get_numpy())
    feats+=list(pca.explained_variance_ratio_)
    for comp_i in pca.components_:
        feats+=list(comp_i)	
    return feats

def corl_features(img,pcloud):
    feats=[]
    points=pcloud.get_numpy()
    def corl_helper(x_i,y_i):
        corr_xy=scipy.stats.pearsonr(points[:,x_i],points[:,y_i])
        return corr_xy[0]
    feats.append(corl_helper(0,1))    
    feats.append(corl_helper(1,2))    
    feats.append(corl_helper(2,0))    
    return feats

def height_feat(img,pcloud):
    img2D=img.get_orginal()
    x,y=img2D.shape
    return [float(x)/float(y)]

def area_feat(img,pcloud):
    points=img[img!=0.0]
    nonzero_points=float(points.shape[0])
    all_points=float(np.prod(img.shape))
    return [nonzero_points/all_points]

class OutlinersExtractor(object):
    def __init__(self,dim=2,neg=False):
        if(dim>2):
            raise Exception("Too high dim %d" % dim)
        self.dim=dim
        self.neg=neg

    def __call__(self,img,pcloud):
        point_selector=self.get_selector(pcloud)
        outliner_pcloud=pcloud.select(point_selector)
        outliner_size=len(outliner_pcloud)
        if(outliner_size==0):
            return [ -1 for  i in range(7)]
        rela_size= self.relative_size(outliner_pcloud,pcloud)
        feat=skewness_features(img,outliner_pcloud)
        feat+=std_features(img,outliner_pcloud)+[rela_size]
        return feat

    def get_selector(self,pcloud):
        points=pcloud.get_numpy()
        all_std=np.std(points,axis=0)
        z_std=all_std[self.dim]
        center=pcloud.center_of_mass()
        z_center=center[self.dim]
        if(self.neg):
            z_std*=(-1.0)
        threshold=z_center+z_std
        if(self.neg):
            return lambda point_i: point_i[self.dim]<threshold 
        else:
            return lambda point_i: point_i[self.dim]>threshold

    def relative_size(self,pcloud_out,pcloud_full):
        return float(len(pcloud_out))/ float(len(pcloud_full))

def elipse_feat(img,pcloud):
    
    raw_img=img.get_orginal()
    int_img=np.uint8(raw_img)
    canny_img=cv2.Canny(int_img,50,150)
    edges = canny_img#utils.canny_transform(canny_img)
    #cv2.Canny(image_gray, sigma=2.0,          low_threshold=0.55, high_threshold=0.8)
    result = cv2.HoughCircles(edges,cv2.HOUGH_GRADIENT,1,20,
  param1=50,param2=30,minRadius=0,maxRadius=0)
    result.sort(order='accumulator')
    best = list(result[-1])
    print(len(best))
    return best

if __name__ == "__main__":
    in_path='../ensemble/full'
    out_path='../ensemble/feat'
    extractor=make_extract()
    action_features(in_path,out_path,extractor)
    #imgs_seq= utils.imgs.make_imgs(in_path,norm=True) 
    #extract_features=make_extract()
    #extract_features(imgs_seq)