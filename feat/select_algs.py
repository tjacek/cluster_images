import numpy as np

def lasso_model(X,y,transform=False):
    clf = linear_model.Lasso(alpha=0.1)
    clf.fit(X,y)
    model = SelectFromModel(clf, prefit=True)
    if(transform):
        return model.transform(X)
    else:
        return model