import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import seq
import seq.dtw
import utils.paths.files

def make_dtw_feat(in_path,out_path,
	              k=0,dataset_format='cp_dataset',select_type='modulo'):
    train,test=seq.seq_dataset(in_path,dataset_format=dataset_format)
    all_seqs=seq.unify_dataset(test,train)
    print(all_seqs.keys())
    seq_xy=get_pairs(all_seqs)
    train_xy=get_pairs(train)

    def dtw_vector(x_i):
        feat_vect=[seq.dtw.dtw_metric(x_i,x_j)
                      for x_j,y_j in train_xy ]
        print(feat_vect)
        print(len(feat_vect))
        return feat_vect

    dtw_feats=[ dtw_vector(x_i) 
                for x_i,y_i in seq_xy]

    def extr_data(i):
        y_i=str(all_seqs['y'][i])
        person_i=str(all_seqs['persons'][i])
        return '#'+y_i+'#'+person_i         
    feat_text= utils.paths.files.seq_to_string(dtw_feats,extr_data)
    utils.paths.files.save_string(out_path,feat_text)

def get_pairs(all_seqs):
    return zip(all_seqs['x'],all_seqs['y'])

if __name__ == "__main__":
    in_path='../../Documents/X2017/dtw_contr/corl_skew/seq'
    out_path='../../Documents/X2017/dtw_contr/corl_skew/dtw_feats.txt'
    make_dtw_feat(in_path,out_path)