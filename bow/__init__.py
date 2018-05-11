import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np
import utils.actions.read
import utils.actions.tools
import bow.hist

def show_separation(in_path,dataset_format='cp_dataset'):
    action_reader=utils.actions.read.ReadActions(dataset_format,img_seq=False,as_dict=False)
    actions=action_reader(in_path)
    quality=compute_quality(actions)
    print(np.sort(quality) )    

def compute_quality(actions):
    hists=bow.hist.make_action_histograms(actions)
    n_feats=utils.actions.tools.count_feats(actions)
    n_actions=len(actions)
    hist_by_feat=[[ hists[j][i]
                    for j in range(n_actions)] 
                        for i in range(n_feats)]
    quality=[feature_quality(hist_by_feat[i])
                for i in range(n_feats)]
    return quality

def feature_quality(histograms,n_cats=20):
    hist_by_cat=bow.hist.hist_by_cat(histograms,n_cats)
    cat_dist=[[ hist_distance(hist_i,hist_by_cat[cat_j])
                for cat_j in range(n_cats)]
                    for hist_i in histograms]
    
    def sep_helper(i,hist_i):
        cat_dist_i= cat_dist[i]
        cat_i= hist_i.info['cat']
        return action_measure(cat_i,cat_dist_i)
    
    for i,hist_i in enumerate(histograms):
        hist_i.info['quality']=sep_helper(i,hist_i)
    return feature_measure(hist_by_cat)    

def show_histogram(hists):
    for i,hist_i in enumerate(hists):
        print(i)
        print(len(hist_i))
        print(hist_i)

def hist_distance(hist_i, hist_list):
    dist_matrix=[ np.linalg.norm(hist_i.data-hist_j.data) 
                    for hist_j in hist_list]
    return np.mean(dist_matrix)
   
def action_measure(cat_i,cat_dist):
    cat_i_dist=cat_dist[cat_i]
    sep_quality =(np.median(cat_dist)-cat_i_dist) #/(1.0 + cat_i_dist)
    return sep_quality

def feature_measure(hist_by_cat):
    quality_matrix=[[hist_ij.info['quality'] 
                        for hist_ij in cat_hist_i]
                            for cat_hist_i in hist_by_cat.values()]  
    
    fm=max([np.median(q_i) for q_i in quality_matrix])
    print(fm)
    return fm

if __name__ == "__main__":
    in_path="../../AA_disk3/clust_seqs/nn_1"	
    show_separation(in_path)
