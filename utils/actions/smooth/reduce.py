import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np 
import utils.actions.smooth

def make_transform(pairs,n=50):
    y=[ pair_i[0] for pair_i in pairs]
    X=[ pair_i[1] for pair_i in pairs]
    svc = SVC(kernel='linear',C=1)
    rfe = RFE(estimator=svc,n_features_to_select=n,step=1)
    rfe.fit(X,y)
    return rfe