import preproc
import seq.features
import utils.conf
import utils.actions

def make_seq_files(conf_path, new_features=True,short_names=True,weight=0.25):
    if(type(conf_path)==str):
        conf_dict=utils.conf.read_config(conf_path)
    else:
        conf_dict=conf_path
    if(new_features):
        preproc.make_features(conf_dict,weight)
    img_path=conf_dict['img_path']
    feat_path=conf_dict['feat_path']
    seq.features.extract_features(img_path,feat_path,'seq',short_names)

def easy_make_seq(conf_path,new_feat=True,weight=1.00):
    print(conf_path)
    make_seq_files(conf_path, new_feat, not new_feat,weight)

if __name__ == "__main__":
    conf_path="conf/1exp1b.cfg"
    easy_make_seq(conf_path, new_feat=True)