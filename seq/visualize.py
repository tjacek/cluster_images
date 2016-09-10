import dtw
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import MDS
import sklearn.manifold as manifold
import utils.data

def visualize_dtw(instances):
    X,y=visualize_metric(instances,dtw.dtw_metric)
    labeled_plot(X,y)
    print(X.shape)    

def visualize_metric(instances,dwt_metric):
    n_samples=len(instances)
    similarities=get_similarity_matrix(n_samples,instances,dwt_metric)       
    X=get_spectral(similarities)
    y=[ seq_i.cat for seq_i in instances]
    return X,y

def get_similarity_matrix(n_samples,instances,dwt_metric):
    similarities=np.zeros((n_samples,n_samples))
    for index, x in np.ndenumerate(similarities):
        x_i,y_i=index
        inst_i=instances[x_i]
        inst_j=instances[y_i]
        similarities[x_i][x_j]=dwt_metric(inst_i,inst_j)
    similarities=norm_x(similarities)
    #similarities=1.0-similarities  
    return similarities

def get_mds(similarities):
    seed = np.random.RandomState(seed=3)
    print(np.amax(similarities))
    print(np.amin(similarities))
    nmds = MDS(n_components=2, metric=False, max_iter=3000, eps=1e-12,
                    dissimilarity="precomputed", random_state=seed, n_jobs=1,
                    n_init=1)
    pos = nmds.fit(similarities).embedding_
    X=np.array(pos)
    return X

def get_spectral(similarities):
    seed = np.random.RandomState(seed=3)
    se = manifold.SpectralEmbedding(n_components=2,
                                n_neighbors=5,affinity="precomputed")
    pos = se.fit(similarities).embedding_
    X=np.array(pos)
    return X

class GetShape(object):
    def __init__(self,colors='bgrcmykw', shape='ovs^p'):
        self.colors=colors
        self.shape=shape

    def __call__(self):
        j=index % len(self.colors)
        s_i=index/len(self.colors)
        s_i=s_i%len( self.shape)
        return self.colors[j],self.shape[s_i]

def labeled_plot(X,y):
    y=np.array(utils.data.to_ints(y),dtype=int)
    get_shape=GetShape()
    fig, ax = plt.subplots()
    x_0=X[:,0]
    x_1=X[:,1]
    n_cats=np.amax(y)+1
    for i in range(n_cats):
        cat_i_0=x_0[y==i]
        cat_i_1=x_1[y==i]
        color_i,shape_i=get_shape(i)
        ax.scatter(cat_i_0,cat_i_1,c=color_i,marker=shape_i,label=str(i))
    plt.legend()
    plt.show()   

def norm_x(X):
    x_max=np.max(X)
    X/=x_max
    return X