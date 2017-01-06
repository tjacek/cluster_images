import numpy as np

class Ensemble(object):
    def __init__(self,simple_cls):
        self.simple_cls=simple_cls

    def get_category(self,action):
        result=[cls_i.gini_weighted(action) 
                 for cls_i in self.simple_cls]
        result=np.array(result)
        dist=np.sum(result,axsis=0)
        return np.argmax(dist)

def make_ensemble(conf):
	simple_cls=[]
    return Ensemble(simple_cls)	