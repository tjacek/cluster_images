import preproc
import seq.features
import utils.conf
import utils.actions

def make_seq_files(conf_path, new_features=True,short_names=True):
    conf_dict=utils.conf.read_config(conf_path)
    if(new_features):
        preproc.make_features(conf_dict,weight=0.25)
    img_path=conf_dict['img_path']
    feat_path=conf_dict['feat_path']
    seq.features.extract_features(img_path,feat_path,'seq',short_names)

def easy_make_seq(conf_path,new_feat=True):
    make_seq_files(conf_path, new_feat, not new_feat)

if __name__ == "__main__":
    conf_path="conf/dane3.cfg"
    easy_make_seq(conf_path, new_feat=False)
