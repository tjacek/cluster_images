import reduce_dim
import clustering
import init_features as init

INIT_FEATURES={'autoencoder':init.use_autoencoder,
               'shape_context':init.use_shape_context,
               'basic':init.use_basic}

REDUCTION={"spectral_reduction":reduce_dim.spectral_reduction,
           "hessian_reduction":reduce_dim.hessian_reduction}

CLUSTER={'kmeans':clustering.kmeans,
         'dbscan':clustering.dbscan,
         'agglomer':clustering.agglomer}
