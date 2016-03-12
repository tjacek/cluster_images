import numpy as np
from sklearn.decomposition import PCA

def pca_features(pcloud):
	feats=[]
	pca = PCA(n_components=3)
    pca.fit(pcloud.get_numpy())
    feats+=pca.explained_variance_ratio_
    for comp_i in pca.components_:
        feats+=list(comp_i)	
    return feats