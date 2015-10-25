from sklearn import manifold

def spectral_reduction(data,dim=20):
    embedder = manifold.SpectralEmbedding(n_components=dim, random_state=0,
                                     eigen_solver="arpack",n_neighbors=5)
    X_prim=embedder.fit_transform(data)
    return X_prim
