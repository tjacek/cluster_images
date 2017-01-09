import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import deep.tools as tools
import deep.convnet
import utils.actions
import numpy as np
import seq.to_dataset
import utils.data 
import ensemble.dispersion

PREPROC_DICT={"time":tools.ImgPreproc2D,
              "proj":tools.ImgPreprocProj}

class SingleCls(object):
    def __init__(self,conv,lstm,max_seq):
        self.conv=conv
        self.lstm=lstm
        self.max_seq=max_seq
    
    def gini_weighted(self,action):
        dist=self(action)
        gini_weight=ensemble.dispersion.gini_index_simple(dist)
        return gini_weight*dist
        
    def __call__(self,action):
        seq_feats=[self.conv(img_i)
                    for img_i in action.img_seq]
        seq_feats_arr=np.array(seq_feats)
        
        seq_size=seq_feats_arr.shape[0]
        seq_dim=seq_feats_arr.shape[1]
        
        masked_feats=masked_seq(seq_feats_arr,self.max_seq,seq_dim)
        mask=make_mask(seq_size,self.max_seq)
        return self.lstm.get_distribution(masked_feats,mask)                	

def make_mask(seq_size,max_size):
    mask=np.zeros((max_size,))
    mask[0:seq_size]=1.0
    return mask

def masked_seq(seq_i,max_seq,seq_dim):
    seq_i_len=utils.data.seq_len(seq_i)
    new_seq_i=np.zeros((max_seq,seq_dim))
    new_seq_i[:seq_i_len]=seq_i[:seq_i_len]
    return new_seq_i

def make_single_cls(conv_path,lstm_path):
    conv=read_convnet(conv_path)
    nn_reader=deep.reader.NNReader()
    lstm,hyperparams=nn_reader(lstm_path,0.5,get_hyper=True)
    return SingleCls(conv,lstm,hyperparams['max_seq'])

def read_actions(cat_path):
    action_reader=utils.actions.ReadActions('cp_dataset')
    actions=action_reader(cat_path)
    return actions

def read_convnet(nn_path,prep_type="time"):
    preproc_method=PREPROC_DICT[prep_type]
    preproc=preproc_method()
    return deep.convnet.get_model(preproc,nn_path,compile=False)

if __name__ == "__main__":
    conv_path='../dataset1/exp1/nn_17'
    #read_convnet(nn_path,"time")
    cat_path='../dataset1/exp1/cats'
    lstm_path='../dataset1/exp1/lstm_full'
    single_cls=make_single_cls(conv_path,lstm_path)
    actions=read_actions(cat_path)
    print(single_cls.gini_weighted(actions[1]) )