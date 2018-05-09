import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np
import utils.actions.read
import utils.actions.tools

def show_separation(in_path,dataset_format='cp_dataset'):
    action_reader=utils.actions.read.ReadActions(dataset_format,img_seq=False,as_dict=False)
    actions=action_reader(in_path)
    by_cat=utils.actions.tools.by_category(actions)
    n_cats=len(by_cat)
    n_feats=utils.actions.tools.count_feats(actions)
    histograms=[make_histograms(actions_i) for actions_i in by_cat.values()]  
    
    hist_by_feat=[[histograms[i][j] 
                    for i in range(n_cats)]
                        for j in range(n_feats)]
    sepa_meas=[cat_separation(hist_of_feat) 
                for hist_of_feat in hist_by_feat]
    for sepa_i in  np.sort(sepa_meas):
        print(sepa_i)
    #print(len(histograms[0][0]))    


def show_histogram(in_path,dataset_format='cp_dataset'):
    action_reader=utils.actions.read.ReadActions(dataset_format,img_seq=False,as_dict=False)
    actions=action_reader(in_path)
    histograms=make_histograms(actions)
    for histogram_i in histograms:
        print(histogram_i)

def cat_separation(hist_of_feat):
    dist_matrix=[[ np.linalg.norm(hist_i-hist_j) 
                    for hist_i in hist_of_feat]
                        for hist_j in hist_of_feat]
    dist_matrix=np.array(dist_matrix)
    return np.sum(dist_matrix)#, axis=0) 
   
def make_histograms(actions,n_clust=10):
    frames=utils.actions.tools.get_frames(actions)
    features=utils.actions.tools.to_features(frames)
    histograms=[get_histogram(feature_i,n_clust) for feature_i in features]
    return histograms

def get_histogram(feature_i,n_clust=10):
    histogram=np.zeros( (n_clust,),dtype=float)
    for x_j in feature_i:
    	j=int(x_j)-1
    	histogram[j]+=1
    histogram/=sum(histogram)
    return histogram

if __name__ == "__main__":
    in_path="../../AA_disk3/clust_seqs/nn_0"	
    show_separation(in_path)
