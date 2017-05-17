import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import seq.dtw
import split
from seq.to_dataset import seq_dataset
import utils.paths.files

def make_dtw_feat(dataset_path,out_path,
	              k=0,dataset_format='cp_dataset',select_type='modulo'):
    train,test,all_seqs=read_seqs(dataset_path,k=k,dataset_format=dataset_format,select_type=select_type)
    
    print(all_seqs.keys())
    seq_xy=zip(all_seqs['x'],all_seqs['y'])

    def dtw_vector(x_i):
        feat_vect=[seq.dtw.dtw_metric(x_i,x_j)
                      for x_j,y_j in seq_xy ]
        print(feat_vect)
        return feat_vect

    dtw_feats=[ (y_i,dtw_vector(x_i)) 
                for x_i,y_i in seq_xy]

    def extr_data(i):
        return '#'+all_seqs['y'][i]+'#'+all_seqs['names'][i]         
    feat_text= utils.paths.files.seq_to_string(dtw_feats,extr_data)
    utils.paths.files.save_string(out_path,feat_text)

def read_seqs(dataset_path,k=0,dataset_format='cp_dataset',select_type='modulo'):
    dataset=seq_dataset(dataset_path,dataset_format)
    split_dataset= split.get_dataset(k,select_type)
    train,test=split_dataset(dataset)
    return train,test,dataset

if __name__ == "__main__":
    #path='../cross/1_set/u_seq'
    path='../inspect/b_nn/seq'
    out_path='../inspect/b_nn/feat.txt'
    make_dtw_feat(path,out_path)