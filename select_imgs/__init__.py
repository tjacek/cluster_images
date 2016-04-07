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


def use_init_features(in_path,config):
    alg=INIT_FEATURES[config['init_features']]
    reduced_data=alg(in_path,config)
    reduced_data=[img_i for img_i in reduced_data
                        if img_i!=None]
    print("init features")
    return reduced_data

def use_reduction(mf_data,config):
    alg_name=config.get('reduce_alg',None)
    if(alg_name==None):
        return mf_data
    reduce_alg=ALGS[alg_name]
    reduced_data=reduce_alg( mf_data,config)
    print("reduce data")
    return reduced_data

def use_clustering(mf_data,config):
    clust_alg=CLUSTER[config['cls_alg']]
    img_cls= clust_alg(mf_data,config) #clustering.dbscan(mf_data,config)
    print("cluster data")
    return img_cls

