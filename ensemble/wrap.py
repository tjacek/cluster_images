import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np
import utils.actions
import ensemble
import ensemble.feat_seq
import ensemble.nn_ensemble
from seq.dtw import dtw_metric

class EnsembleDTW(object):
    def __init__(self,single_cls,datasets):
        self.single_cls=single_cls
        self.datasets=datasets

    def get_category(self,action_x):
        results=[ cls_i(self.get_action(type_i,action_x)) 
                  for type_i,cls_i in self.single_cls.items()]
        #print(results)
        cats_k=[ cat_i[0] for cat_i,dist_i in results ]
        print(cats_k)
        dist_k=[ dist_i[0] for cat_i,dist_i in results]          
        min_i=np.argmin(dist_k)
        return cats_k[min_i]

    def get_action(self,type_i,action_i):
        return self.datasets[type_i][action_i.name]
    
    def __len__(self):
        return sum( [len(cls_i)  
                     for type_i,cls_i in self.single_cls.items()])

class SingleDTW(object):
    def __init__(self,conv,actions,cats,scale=1.0):
        if(len(actions)==0):
            raise Exception("No dataset for dtw")
        self.actions=actions
        self.cats=cats
        self.conv=conv
        self.scale=scale

    def __call__(self,action,k=5):
        action_feat=self.conv(action)
        action_feat=[self.scale * frame_i 
                       for frame_i in action_feat]
        distance=[dtw_metric(action_feat,action_i) 
                    for action_i in self.actions]
        distance=np.array(distance)
        dist_inds=distance.argsort()[0:k]
        #distance_k=distance[dist_inds]
        distance_k=[distance[i] * self.scale for i in dist_inds]
        #print(distance_k.dtype)
        #print(distance_k.shape)
        cats_k =[ int(self.cats[i])-1
                  for i in dist_inds]
        print("########################")
        print(distance_k)
        print(cats_k)
        print("########################")
        return cats_k,distance_k
    
    def get_category(self,action_x):
        return self(action_x)[0][0] 

    def __len__(self):
        return len(self.actions)

    def dataset_scale(self):
        pair_dist=[dtw_metric(action_i,action_j)
            for action_i in self.actions
              for action_j in self.actions]
        return np.array(pair_dist).mean()

    def dataset_size(self):
        size=0.0
        for i,action_i in enumerate(self.actions):
            print(i)
            for frame_i in action_i:
                size+=np.linalg.norm(frame_i,ord=2)
        return size

def make_ensemble_dtw(dataset_paths,nn_paths,scales,prep_type="time"):
    datasets=ensemble.nn_ensemble.make_datasets(dataset_paths,s_action=None)    
    def get_single(i,type_i):
        nn_path_i=nn_paths[type_i]
        dataset_i=datasets[type_i]

        scale_i=scales[i]
        return make_single_dtw(nn_path_i,dataset_i,scale_i,prep_type=type_i) 
    single_cls={ type_i:get_single(i,type_i)
                  for i,type_i in enumerate(dataset_paths.keys())}
                  #for nn_path_i,scale_i in zip(nn_paths,scales)]
    return EnsembleDTW(single_cls,datasets)                

def make_single_dtw(nn_path,dataset,scale,prep_type="time"):
    conv=ensemble.feat_seq.make_feat_seq(nn_path,prep_type)
    s_dataset=select_dataset(dataset)
    actions=s_dataset.values()
    conv_actions=[conv(action_i) 
                   for action_i in actions]
    cats=[ action_i.cat
            for action_i in actions]
    return SingleDTW(conv,conv_actions,cats,scale)

def select_dataset(dataset):
    actions=dataset.values()
    if(len(actions)==0):
        raise Exception("No dataset")
    s_actions=utils.actions.select_actions(actions,action_type='odd')
    if(len(s_actions)==0):
        raise Exception("Wrong selection")
    return { action_i.name:action_i
             for action_i in s_actions}

def get_weights(ensemble_DTW):
    raws=[cls_i.dataset_size() 
            for cls_i in ensemble_DTW.single_cls.values()]
    max_i=max(raws)
    weights=[ raw_i/max_i 
              for raw_i in raws]
    return weights

if __name__ == "__main__":
    nn_paths={ 'time':"../dataset1/exp1/nn_data_1",
               'proj':'../dataset1/exp2/old/nn_worst'}
    dataset_paths={'time':'../dataset1/exp1/full_dataset',
                   'proj':'../dataset1/exp2/cats'}
    scales=[1.00,0.64]
    dataset_path="../dataset1/exp1/full_dataset"
    ensemble_cls=make_ensemble_dtw(dataset_paths,nn_paths,scales)
    ensemble.test_model(ensemble_cls,dataset_path,action_selection='even')
    #print(get_weights(ensemble_cls))

    #dataset=ensemble.nn_ensemble.make_actions_dict(dataset_paths['time'],s_action='odd')    
    #single=make_single_dtw(nn_paths['time'],dataset,scale=1.0,prep_type="time")
    #print(single.dataset_size())