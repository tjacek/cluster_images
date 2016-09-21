import cv2
import numpy as np
import utils.pcloud as pcloud
import scipy.stats
import scipy.stats.stats as st
from sklearn.decomposition import PCA
import utils

def get_features(img):
    print(img.shape)
    points=pcloud.make_point_cloud(img)
    #points=pcloud.unit_normalized(points)
    points=pcloud.normalized_cloud(points)
    if(points==None):
    	return None
    cloud_extractors=[area_feat,skewness_features,center,std_features,
                      extr_features]#,elipse_feat]
    all_feats=[]
    for extr_i in cloud_extractors:
        all_feats+=extr_i(img,points)
    print(all_feats)      	
    return np.array(all_feats)

def extr_features(img,pcloud):
    extr=list(pcloud.min(2))
    extr+=list(pcloud.max(2))
    print(len(extr))
    return extr

def center(img,pcloud):
    return list(pcloud.center_of_mass())

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
    dim=pcloud.dims
    for x_i in range(dim):
        for y_i in range(dim):
            if(x_i!=y_i):
                corr_xy=scipy.stats.pearsonr(points[:,x_i],points[:,y_i])
                feats.append(corr_xy[0])    
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