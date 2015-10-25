from sklearn import manifold

def spectral_reduction(data,dim=20):
    embedder = manifold.SpectralEmbedding(n_components=dim, random_state=0,
                                     eigen_solver="arpack",n_neighbors=20)
    X_prim=embedder.fit_transform(data)
    return X_prim

def hessian_reduction(data,dim=12,n_neighbors=100):
    lle= manifold.LocallyLinearEmbedding(n_neighbors, n_components=dim,
                                      method='hessian',eigen_solver="dense")
    X_prim=lle.fit_transform(data)
    return X_prim
