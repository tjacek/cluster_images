import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np 
import utils.actions.smooth

class ReduceActions(utils.actions.smooth.TimeSeriesTransform):
    
    def get_series_transform(self,frames):
        y=[ pair_i[0] for pair_i in pairs]
        X=[ pair_i[1] for pair_i in pairs]
        svc = SVC(kernel='linear',C=1)
        rfe = RFE(estimator=svc,n_features_to_select=n,step=1)
        rfe.fit(X,y)
        def redu_transform(x):
            x=x.reshape(1, -1)
            new_frame=rfe.transform(x)
            return new_frame.flatten()
        return redu_transform