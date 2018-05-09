import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np
import utils.actions.read
import utils.actions.tools

def show_histogram(in_path,dataset_format='cp_dataset'):
    action_reader=utils.actions.read.ReadActions(dataset_format,img_seq=False,as_dict=False)
    actions=action_reader(in_path)
    frames=utils.actions.tools.get_frames(actions)
    features=utils.actions.tools.to_features(frames)
    histograms=[get_histogram(feature_i) for feature_i in features]
    for histogram_i in histograms:
        print(histogram_i)

def get_histogram(feature_i):
    feature_i=np.array(feature_i)
    n_clust=int(np.amax(feature_i))
    if(n_clust==0):
        return None
    histogram=np.zeros( (n_clust,),dtype=float)
    for x_j in feature_i:
    	j=int(x_j)-1
    	histogram[j]+=1
    histogram/=sum(histogram)
    return histogram

if __name__ == "__main__":
    in_path="../../AA_disk3/clust_seqs/nn_0"	
    show_histogram(in_path)
