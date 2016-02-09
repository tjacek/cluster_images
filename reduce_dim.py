from sklearn import manifold

def spectral_reduction(data,config):
    dim=int(config.get('dim',30))
    neighbors=int(config.get('neighbors',40))
    embedder = manifold.SpectralEmbedding(n_components=dim, random_state=0,
    	eigen_solver="arpack",n_neighbors=neighbors)
    X_prim=embedder.fit_transform(data)
    return X_prim

def hessian_reduction(data,config):
    dim=int(config.get('dim',12))
    n_neighbors=config.get('n_neighbors',100)
    lle= manifold.LocallyLinearEmbedding(n_neighbors, n_components=dim,
                                      method='hessian',eigen_solver="dense")
    X_prim=lle.fit_transform(data)
    return X_prim
