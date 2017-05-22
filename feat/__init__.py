import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np
import utils.actions,utils.actions.read
import basic.external
import feat.select_algs

#def reduce_feat(build_path,in_path=None,out_path=None):
#    X_build,y_build,names_build=get_dataset(build_path)
#    model=feat.select_algs.LinearModel(X_build,y_build)
#    if(in_path==None):
#        X_in=X_build
#        names_in=names_build
#    else:
#        X_in,y_in,names_in=get_dataset(in_path)
#    new_X=model(X)
#    to_feat_dict=ToFeatDict()
#    new_feat_dict=to_feat_dict(new_X,names_in)
#    if(out_path!=None):
#        new_feat_dict.save(out_path)
#    return new_feat_dict

#def get_dataset(in_path):
#    feat_dict=basic.external.read_external(in_path)
#    to_dataset=ToDataset()
#    X,y,names=to_dataset(feat_dict)
#    return X,y,names

def reduce_feat(in_path,model,out_path=None):
    feat_dict=basic.external.read_external(in_path)
    to_dataset=ToDataset()
    X,y,names=to_dataset(feat_dict)
    new_X=model(X,y,True)
    print("New shape %d " % new_X.shape[1])
    to_feat_dict=ToFeatDict()
    new_feat_dict=to_feat_dict(new_X,names)
    if(out_path!=None):
        new_feat_dict.save(out_path)
    return new_feat_dict

class ToFeatDict(object):
    def __init__(self,  dataset_format=None):
        self.dataset_format=default_format(dataset_format)
        
    def __call__(self,X,names): 
        raw_dict={ name_i:X[i]
                  for i,name_i in enumerate(names)}
        return basic.external.ExternalFeats( raw_dict)

class ToDataset(object):
    def __init__(self, dataset_format=None):
        self.dataset_format=default_format(dataset_format)
    
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

def unify_feat(in_paths,out_path=None):
    feat_dict=[basic.external.read_external(in_path_i)
                 for in_path_i in in_paths]
    keys=feat_dict[0].names()
    def unify_helper(name_i):
        vectors=[feat_dict_i[name_i] 
                  for feat_dict_i in feat_dict]
        return np.concatenate(vectors)
    unified_dict={ key_i:unify_helper(key_i)
                   for key_i in keys}
    unified_dict= basic.external.ExternalFeats(unified_dict)
    if(out_path!=None):
        unified_dict.save(out_path)
    return unified_dict                   

def feat_to_actions(in_path,out_path,dataset_format='cp_dataset'):
    feat_dict=basic.external.read_external(in_path)
    action_feat(feat_dict,out_path,dataset_format)

def action_feat(feat_dict,out_path,dataset_format='cp_dataset'):
    action_dict=feat_dict.divided_by_action()
    new_actions=utils.actions.read.NewActions(dataset_format)
    actions=new_actions(action_dict)
    save_actions=utils.actions.read.SaveActions(img_actions=False)
    save_actions(actions,out_path)

def default_format(dataset_format):
    if(dataset_format==None):
        return utils.actions.read.cp_dataset
    else:
        return dataset_format
    
if __name__ == "__main__":
    build_path='../ensemble3/_nn/build_feat.txt'
    in_path='../ensemble3/basic_nn/feat.txt'
    out_path='../ensemble3/easy_nn/seq2'
    #paths=['../ensemble3/10_nn/feat2.txt','../ensemble3/17_nn/feat2.txt']

    paths=[ '../inspect/combine/feat.txt', 
               '../inspect/combine/feat2.txt'] 
    #new_dict=reduce_feat(in_path,feat.select_algs.lasso_model,out_path='../ensemble3/basic_nn/feat2.txt')
    #action_feat(new_dict,out_path)
    
    #unify_feat(paths,out_path='../inspect/combine/u_feat.txt')
    feat_to_actions('../dtw_feat/simple4/feat.txt','../dtw_feat/simple4/seq')
    