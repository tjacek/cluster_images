import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn import linear_model

def lasso_model(X,y,transform=False):
    clf = linear_model.Lasso(alpha=0.1)
    clf.fit(X,y)
    model = SelectFromModel(clf, prefit=True)
    if(transform):
        return model.transform(X)
    else:
        return model

def tree_model(X,y,transform=False):
    clf = linear_model.ExtraTreesClassifier()
    clf.fit(X,y)
    model = SelectFromModel(clf, prefit=True)
    if(transform):
        return model.transform(X)
    else:
        return model