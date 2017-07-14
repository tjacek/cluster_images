import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import preproc
import utils.paths
import seq.features,seq.dtw_feat

def make_feats(img_path,feat_path,seq_path,dataset_format='cp_dataset'):
    #seq_path=get_seq_path(feat_path,dir_name='seq')
    #feat_path=get_seq_path(nn_path,dir_name='feat')
    conf={'img_path':img_path,'nn_path':feat_path,'feat_path':feat_path,'preproc':'proj','extractor':'deep_dist'}
    preproc.make_features(conf,1.0)
    seq.features.extract_features(img_path,feat_path,seq_path,False)
    #if(out_path!=None):
    #    seq.dtw_feat.make_dtw_feat(seq_path,out_path,dataset_format=dataset_format)

def get_seq_path(input_path='nn_path',dir_name='seq'):
    raw_path=utils.paths.Path(input_path)
    return raw_path.copy().set_name(dir_name)

if __name__ == "__main__":
    #img_path="../../MSRA/time"#old/nn_recr"
    #nn_path="../../exper2/time/nn_time"
    img_path='../../final_paper/MSRaction/full'
    nn_path='../../final_paper/MSRaction/basic_nn/nn_basic'
    out_path="../../konf/seq"
    make_feats(img_path,nn_path,out_path)