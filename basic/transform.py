import numpy as np
from sklearn.preprocessing import robust_scale

def scale_features(feat_dict):
    return transform_features(feat_dict,robust_scale)

def transform_features(feat_dict,transform):
    index_dict,features=dict_to_numpy(feat_dict)
    new_features=transform(features)
    return update_dict(feat_dict,index_dict,new_features)

def dict_to_numpy(feat_dict):
    index_dict={}
    features=[]
    for i,key_i in enumerate(feat_dict.keys()):
        index_dict[key_i]=i
        features.append(feat_dict[key_i])
    return index_dict,np.array(features,dtype=float)

def update_dict(feat_dict,index_dict,features):
    new_dict={}
    for key_i in feat_dict.keys():
        new_value=features[index_dict[key_i]]
        new_dict[key_i]=new_value
    return new_dict