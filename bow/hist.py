import numpy as np
import utils.actions.tools

class Histogram(object):
    def __init__(self,data,feature,info=None):
        self.feature=feature
        self.info=info    
        self.data=data

    def __getitem__(self,i):
        return self.data[i]

def make_action_histograms(actions,n_clust=10):
    def action_helper(j,action_i,feature_j):
        hist_ij=get_histogram(feature_j,n_clust=10)
        cat_i=int(action_i.cat)-1
        info={'cat':cat_i,'name':action_i.name}
        return Histogram(hist_ij,j,info)
    return [[  action_helper(j,action_i,feature_j)
                for j,feature_j in enumerate(action_i.to_series())] 
                    for action_i in actions]

def make_cat_histograms(actions,n_clust=10):
    by_cat=utils.actions.tools.by_category(actions)
    n_cats=len(by_cat)
    n_feats=utils.actions.tools.count_feats(actions)
    def cat_helper(cat_i,cat_i_actions):
        hists=make_feat_histograms(cat_i_actions_i,n_clust)
        for his_i in hists:
            his_i.info={'cat': cat_i}
        return hists 
    histograms={ cat_i:cat_helper(cat_i,cat_i_actions)
                    for cat_i,cat_i_actions in by_cat.items()}  
    return histograms

def make_feat_histograms(actions,n_clust=10):
    frames=utils.actions.tools.get_frames(actions)
    features=utils.actions.tools.to_features(frames)
    def feat_helper(i,feature_i):
        raw_hist=get_histogram(feature_i,n_clust)
        return Histogram(raw_hist,i)
    histograms=[ feat_helper(i,feature_i)
                    for i,feature_i in enumerate(features)]
    return histograms

def get_histogram(feature_i,n_clust=10):
    histogram=np.zeros( (n_clust,),dtype=float)
    for x_j in feature_i:
    	j=int(x_j)-1
    	histogram[j]+=1
    histogram/=sum(histogram)
    return histogram

def hist_by_cat(histograms,n_cats=20):
    by_cat={ cat_i:[] for cat_i in range(n_cats)}
    for hist_i in histograms:
        cat_i=hist_i.info['cat']
        by_cat[cat_i].append(hist_i)
    return by_cat	