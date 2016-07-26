import numpy as np
import sklearn.manifold
from sklearn import decomposition

class TransformSpectral(object):
    def __init__(self,dim=30,neighbors=20):
        self.dim=dim
        self.n_neighbors=n_neighbors

    def __call__(self,data):
        X=np.array(data)
        embedder = sklearn.manifold.SpectralEmbedding(
                    n_components=self.dim, random_state=0,
                    eigen_solver="arpack",n_neighbors=self.n_neighbors)
        X_prim=embedder.fit_transform(X)
        return X_prim

class TransformPca(object):
    def __init__(self, dim=30):
        self.dim=dim
        
    def __call__(self,data):
        X=np.array(data)
        pca = decomposition.SparsePCA(n_components=self.dim)
        pca.fit(X)
        X_prim = pca.transform(X)
        return X_prim
