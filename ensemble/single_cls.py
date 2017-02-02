import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import utils.actions
import numpy as np
import seq.to_dataset
import utils.data 
import ensemble.feat_seq
import ensemble.dispersion
import deep.reader
import utils.actions
import ensemble

class SingleCls(object):
    def __init__(self,feat_seq,lstm,max_seq,seq_dim):
        self.feat_seq=feat_seq
        self.lstm=lstm
        self.max_seq=max_seq
        self.seq_dim=seq_dim

    def gini_weighted(self,action):
        dist=self(action)
        gini_weight=ensemble.dispersion.gini_index_simple(dist)
        return gini_weight*dist
        
    def __call__(self,action):
        seq_feats=self.feat_seq(action)
        seq_len=len(seq_feats)
        #seq_dim=seq_feats[0].shape[0]
        seq_diff=self.max_seq-seq_len
        for i in range(seq_diff):
            seq_feats.append(np.zeros((self.seq_dim,)))
        masked_feats=np.array(seq_feats)
        mask=make_mask(seq_len,self.max_seq)
        return self.lstm.get_distribution(masked_feats,mask)
        #seq_feats_arr=np.array(seq_feats)
        
        #seq_size=seq_feats_arr.shape[0]
        #seq_dim=seq_feats_arr.shape[1]
        
        #print(seq_size)
        #print(seq_dim)
        #masked_feats=masked_seq(seq_feats_arr,seq_size,self.max_seq,seq_dim)
        #mask=make_mask(seq_size,self.max_seq)
        #result=self.lstm.get_distribution(masked_feats,mask)                	
        #return result

    def get_category(self,action):
        return np.argmax(self(action))

def make_mask(seq_size,max_size):
    mask=np.zeros((max_size,),dtype=float)
    mask[:seq_size]=1.0
    return mask

def masked_seq(seq_i,seq_i_len,max_seq,seq_dim):
    #seq_i_len=utils.data.seq_len(seq_i)

    new_seq_i=np.zeros((max_seq,seq_dim))
    new_seq_i[:seq_i_len]=seq_i[:seq_i_len]
    return new_seq_i

def make_single_cls(conv_path,lstm_path,with_type=False,prep_type="time"):
    seq_feat=ensemble.feat_seq.make_feat_seq(conv_path,prep_type)
    nn_reader=deep.reader.NNReader(deep.reader.get_preproc(prep_type))
    lstm,hyperparams=nn_reader(lstm_path,drop_p=0.0,get_hyper=True)
    single_cls=SingleCls(seq_feat,lstm,hyperparams['max_seq'],hyperparams['seq_dim'])
    if(with_type):
        return with_type,single_cls
    return single_cls

if __name__ == "__main__":
    conv_path='../dataset1/exp2/nn_worst'
    cat_path='../dataset1/exp2/cats'
    lstm_path='../dataset1/exp2/lstm_worst'
    single_cls=make_single_cls(conv_path,lstm_path,prep_type='proj')
    actions=ensemble.read_actions(cat_path)
    s_actions= utils.actions.select_actions(actions)
    ensemble.check_model(single_cls,s_actions)