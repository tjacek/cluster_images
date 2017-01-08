import numpy as np
import single_cls as s_cls

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
    assert(type(conf)==list)
    simple_cls=[s_cls.make_single_cls(conf_i[1],conf_i[0])
                  for conf_i in conf]
    return Ensemble(simple_cls)

if __name__ == "__main__":
    full=('../dataset1/exp1/lstm_full','../dataset1/exp1/nn_full')
    f_4=('../dataset1/exp1/lstm_4','../dataset1/exp1/nn_4')
    ens=make_ensemble([full,f_4])