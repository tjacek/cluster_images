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
    def __init__(self,feat_seq,lstm,max_seq,seq_dim,disp=False):
        self.feat_seq=feat_seq
        self.lstm=lstm
        self.max_seq=max_seq
        self.seq_dim=seq_dim
        self.disp=disp

    def __call__(self,action):
        seq_feats=self.feat_seq(action)
        mask,masked_feats=self.get_masked_seq(seq_feats)
        dist= self.lstm.get_distribution(masked_feats,mask)
        if(self.disp):
            quality_factor = np.linalg.norm(dist)
            cat=np.argmax(dist)
            print("quality %f %d" % quality_factor,cat)
            dist=quality_factor*dist
        return dist

    def get_category(self,action):
        return np.argmax(self(action))

    def get_masked_seq(self,seq_feats):
        seq_len=len(seq_feats)
        seq_diff=self.max_seq-seq_len
        for i in range(seq_diff):
            seq_feats.append(np.zeros((self.seq_dim,)))
        masked_feats=np.array(seq_feats)
        mask=make_mask(seq_len,self.max_seq)
        return mask,masked_feats

def make_mask(seq_size,max_size):
    mask=np.zeros((max_size,),dtype=float)
    mask[:seq_size]=1.0
    return mask

def masked_seq(seq_i,seq_i_len,max_seq,seq_dim):
    #seq_i_len=utils.data.seq_len(seq_i)

    new_seq_i=np.zeros((max_seq,seq_dim))
    new_seq_i[:seq_i_len]=seq_i[:seq_i_len]
    return new_seq_i

def make_single_cls(conv_path,lstm_path,with_type=False,prep_type="proj",disp=False):
    seq_feat=ensemble.feat_seq.make_feat_seq(conv_path,prep_type)
    nn_reader=deep.reader.NNReader(deep.reader.get_preproc(prep_type))
    lstm,hyperparams=nn_reader(lstm_path,drop_p=0.0,get_hyper=True)
    single_cls=SingleCls(seq_feat,lstm,hyperparams['max_seq'],hyperparams['seq_dim'],disp)
    if(with_type):
        return with_type,single_cls
    return single_cls

if __name__ == "__main__":
    conv_path='../dataset1/exp1/nn_data_1'
    cat_path='../dataset1/exp1/full_dataset'
    #lstm_path='../dataset1/exp2/lstm_worst'
    #single_cls=make_single_cls(conv_path,lstm_path,prep_type='proj')
    actions=ensemble.read_actions(cat_path)
    print(actions[0].name)
    seq_feat=ensemble.feat_seq.make_feat_seq(conv_path,prep_type='time')
    print(seq_feat(actions[0]))
    #s_actions= utils.actions.select_actions(actions)
    #ensemble.check_model(single_cls,s_actions)