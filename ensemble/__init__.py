import numpy as np
import single_cls as s_cls
import utils.actions
import seq

class Ensemble(object):
    def __init__(self,simple_cls):
        self.simple_cls=simple_cls

    def get_category(self,action):
        result=[cls_i(action) 
                 for cls_i in self.simple_cls]
        result=np.array(result)
        dist=np.sum(result,axis=0)
        return dist.argmax()

def make_ensemble(conf):
    assert(type(conf)==list)
    simple_cls=[s_cls.make_single_cls(conf_i[1],conf_i[0])
                  for conf_i in conf]
    return Ensemble(simple_cls)

def check_model(model,actions):
    print(type(actions[0]))	
    y_true=[int(action_i.cat)-1 for action_i in actions]
    y_pred=[model.get_category(action_i)
              for action_i in actions]
    #print(utils.data.find_errors(y_pred,test))
    seq.check_prediction(y_pred,y_true)

def test_model(model,action_path):
    actions=read_actions(action_path)
    s_actions= utils.actions.select_actions(actions)
    check_model(model,s_actions)

def read_actions(cat_path,action_type='cp_dataset'):
    action_reader=utils.actions.ReadActions(action_type)
    actions=action_reader(cat_path)
    if(len(actions)==0):
        raise Exception("No actions found in: "+ cat_path)
    return actions

if __name__ == "__main__":
    full=('../dataset1/exp1/lstm_full','../dataset1/exp1/nn_full')
    f_4=('../dataset1/exp1/lstm_4','../dataset1/exp1/nn_4')
    trivial=('../dataset1/exp1/lstm_trivial','../dataset1/exp1/nn_trivial')
    ens=make_ensemble([f_4,trivial])
    in_path="../dataset1/exp1/full_dataset"
    read_actions= utils.actions.ReadActions( utils.actions.cp_dataset,norm=True)
    actions=read_actions(in_path)
    s_actions= utils.actions.select_actions(actions)
    check_model(ens,s_actions)