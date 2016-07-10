import numpy as np
import utils.pcloud as pcloud
import scipy.stats
import scipy.stats.stats as st
from sklearn.decomposition import PCA



def get_features(img):
    print(img.shape)
    points=pcloud.make_point_cloud(img)
    points=pcloud.normalized_cloud(points)

    if(points==None):
    	return None
    cloud_extractors=[area_feat,skewness_features,center]#,std_features,skewness_features]
    all_feats=[]
    for extr_i in cloud_extractors:
        all_feats+=extr_i(img,points)
    print(all_feats)      	
    return np.array(all_feats)

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
    x,y=img.shape
    return [float(x)/float(y)]

def area_feat(img,pcloud):
    points=img[img!=0.0]
    nonzero_points=float(points.shape[0])
    all_points=float(np.prod(img.shape))
    return [10.0*nonzero_points/all_points]