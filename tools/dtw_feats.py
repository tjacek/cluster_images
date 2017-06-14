import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import preproc
import feat
import seq.dtw_feat 
import seq,seq.features
import utils.paths

def make_dtw_feats(img_path,nn_path,out_path,dataset_format='cp_dataset'):
    seq_path=get_seq_path(nn_path,dir_name='seq')
    feat_path=get_seq_path(nn_path,dir_name='feat')
    conf={'img_path':img_path,'nn_path':nn_path,'feat_path':feat_path,'preproc':'proj','extractor':'deep'}
    preproc.make_features(conf,1.0)
    seq.features.extract_features(img_path,feat_path,seq_path,False)
    seq.dtw_feat.make_dtw_feat(seq_path,out_path,dataset_format=dataset_format)
    #feat.global.get_global_features(in_path,out_path,dataset_format=dataset_format)

def get_seq_path(input_path='nn_path',dir_name='seq'):
    raw_path=utils.paths.Path(input_path)
    return raw_path.copy().set_name(dir_name)

if __name__ == "__main__":
    img_path= "../../exper/unified/full"
    feat_path="../../exper/unified/nn_basic"
    out_path="../../exper/unified/dataset.txt"
    make_dtw_feats(img_path,feat_path,out_path)
