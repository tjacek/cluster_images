import numpy as np
import utils.pcloud as pcloud
from sklearn.decomposition import PCA

def get_features(img):
    pcloud=make_point_cloud(img)
    extractors=[]
    all_feats=[]
    for extr_i in extractors:
        all_feats+=extr_i(pcloud)    	
    return all_feats

def pca_features(pcloud):
	feats=[]
	pca = PCA(n_components=3)
    pca.fit(pcloud.get_numpy())
    feats+=pca.explained_variance_ratio_
    for comp_i in pca.components_:
        feats+=list(comp_i)	
    return feats