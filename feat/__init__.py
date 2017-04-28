import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np
import utils.actions,utils.actions.read
import basic.external
from sklearn.feature_selection import SelectFromModel
from sklearn import linear_model

def reduce_feat(in_path,model):
    feat_dict=basic.external.read_external(in_path)
    to_dataset=ToDataset()
    X,y,names=to_dataset(feat_dict)
    new_X=model(X,y,True)
    print(new_X.shape)
    to_feat_dict=ToFeatDict()
    new_feat_dict=to_feat_dict(new_X,names)
    return new_feat_dict

def lasso_model(X,y,transform=False):
    clf = linear_model.Lasso(alpha=0.1)
    clf.fit(X,y)
    model = SelectFromModel(clf, prefit=True)
    if(transform):
        return model.transform(X)
    else:
        return model

class ToFeatDict(object):
    def __init__(self,  dataset_format=None):
        if(dataset_format==None):
            self.dataset_format=utils.actions.read.cp_dataset
        else:
            self.dataset_format=dataset_format
        
    def __call__(self,X,names):
        raw_dict={ name_i:X[i]
                  for i,name_i in enumerate(names)}
        return basic.external.ExternalFeats( raw_dict)

class ToDataset(object):
    def __init__(self, dataset_format=None):
        if(dataset_format==None):
            self.dataset_format=utils.actions.read.cp_dataset
        else:
            self.dataset_format=dataset_format

    def __call__(self,feat_dir):
        X=[]
        y=[]
        names=[]
        for key_i,value_i in feat_dir.items(True):
            print(key_i)
            action_i = feat_dir.extract_action(key_i)
            name,cat,person=self.dataset_format(action_i)
            X.append(value_i)
            y.append(int(cat))
            names.append(str(key_i))
        X=np.array(X)
        return X,y,names

def action_feat(feat_dict,out_path,dataset_format='cp_dataset'):
    action_dict=feat_dict.divided_by_action()
    new_actions=utils.actions.read.NewActions(dataset_format)
    actions=new_actions(action_dict)
    save_actions=utils.actions.read.SaveActions(img_actions=False)
    save_actions(actions,out_path)
    
if __name__ == "__main__":
    in_path='../ensemble/hard_nn/feat.txt'
    out_path='../ensemble/hard_nn/seq2'
    new_dict=reduce_feat(in_path,lasso_model)
    action_feat(new_dict,out_path)
    #feat_dict=basic.external.read_external(in_path)
    #to_dataset=ToDataset()
    #X,y,names=to_dataset(feat_dict)

    #to_feat_dict=ToFeatDict()
    #new_feat_dict=to_feat_dict(X,names)

    #key_1=feat_dict.raw_dict.keys()[0]
    #transform_feat(in_path,out_path)
    #print(feat_dict.divided_by_action())