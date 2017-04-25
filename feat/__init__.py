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

def to_dataset(pairs):
    def data_helper(k):
        return [ pair_i[k] 
                 for pair_i in pairs]
    y= data_helper(0)
    X= data_helper(1)
    X=np.array(X)
    return X,y	

if __name__ == "__main__":
    in_path='../ensemble/basic_nn/feat.txt'
    out_path='../ensemble/basic_nn/seq2'
    feat_dict=basic.external.read_external(in_path)
    action_dict=feat_dict.divided_by_action()
    new_actions=utils.actions.read.NewActions('cp_dataset')
    actions=new_actions(action_dict)
    save_actions=utils.actions.read.SaveActions(img_actions=False)
    save_actions(actions,out_path)


    #print(feat_dict.divided_by_action())