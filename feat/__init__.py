import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np
import utils.actions,utils.actions.read
import basic.external
from sklearn.feature_selection import SelectFromModel

def lasso_model(X,y):
    clf = linear_model.Lasso(alpha=0.1)
    clf.fit(X,y)
    #lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
    model = SelectFromModel(clf, prefit=True)
    return model
#    X_new = model.transform(X)

def action_pairs(actions):
    pairs=[]
    for i,action_i in enumerate(actions):
        pairs+=action_i.to_pairs(i)
    return pairs

class ToFeatDict(object):
    def __init__(self,  dataset_format=None):
        if(dataset_format==None):
            self.dataset_format=utils.actions.read.cp_dataset
        else:
            self.dataset_format=dataset_format
        
    def __call__(self,X,names):
        return { name_i:X[i]
                  for i,name_i in enumerate(names)}

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

def transform_feat(in_path,out_path,dataset_format='cp_dataset'):
    feat_dict=basic.external.read_external(in_path)
    action_dict=feat_dict.divided_by_action()
    new_actions=utils.actions.read.NewActions(dataset_format)
    actions=new_actions(action_dict)
    save_actions=utils.actions.read.SaveActions(img_actions=False)
    save_actions(actions,out_path)

if __name__ == "__main__":
    in_path='../ensemble/basic_nn/feat.txt'
    out_path='../ensemble/basic_nn/seq2'
    feat_dict=basic.external.read_external(in_path)
    to_dataset=ToDataset()
    X,y,names=to_dataset(feat_dict)

    to_feat_dict=ToFeatDict()
    new_feat_dict=to_feat_dict(X,names)

    key_1=feat_dict.raw_dict.keys()[0]
    print(feat_dict[key_1])
    print("$$$$$$$$$$$$$$$$$$$$$$$$")
    print(new_feat_dict[key_1])
    #transform_feat(in_path,out_path)
    #print(feat_dict.divided_by_action())