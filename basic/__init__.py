import numpy as np
import utils.pcloud as pcloud
from sklearn.decomposition import PCA

def get_features(img):
    points=pcloud.make_point_cloud(img)
    extractors=[pca_features]
    all_feats=[]
    for extr_i in extractors:
        all_feats+=extr_i(points)
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