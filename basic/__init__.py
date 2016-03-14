import numpy as np
import utils.pcloud as pcloud
import scipy.stats
from sklearn.decomposition import PCA

def get_features(img):
    points=pcloud.make_point_cloud(img)
    if(points==None):
    	return None
    cloud_extractors=[corl_features]
    all_feats=[]
    for extr_i in cloud_extractors:
        all_feats+=extr_i(points)
    img_extractors=[height_feat]
    for extr_i in img_extractors:
        all_feats+=extr_i(img)
    print(all_feats)      	
    return np.array(all_feats)

def pca_features(pcloud):
    feats=[]
    pca = PCA(n_components=3)
    pca.fit(pcloud.get_numpy())
    feats+=list(pca.explained_variance_ratio_)
    for comp_i in pca.components_:
        feats+=list(comp_i)	
    return feats

def corl_features(pcloud):
    feats=[]
    points=pcloud.get_numpy()
    dim=pcloud.dims
    print(dim)
    for x_i in range(dim):
        for y_i in range(dim):
            if(x_i!=y_i):
                corr_xy=scipy.stats.pearsonr(points[x_i,:],points[y_i,:])
                feats.append(corr_xy[0])    
    print(feats)
    return feats

def height_feat(img):
    x,y=img.shape
    return [float(x)/float(y)]