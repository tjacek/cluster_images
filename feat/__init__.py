import numpy as np
import utils.actions
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
    for action_i in actions:
        pairs+=action_i.to_pairs()
    return pairs

def to_dataset(pairs):
    y=[ pair_i[0] 
        for pair_i in pairs]
    X=[ pair_i[1] 
        for pair_i in pairs]
    X=np.array(X)
    return X,y	

if __name__ == "__main__":

