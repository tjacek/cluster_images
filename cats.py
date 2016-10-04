import preproc
import seq.features
import utils.conf
import utils.actions

def make_seq_files(conf_path, new_features=True):
    conf_dict=utils.conf.read_config(conf_path)
    if(new_features):
        preproc.make_features(conf_dict)
    img_path=conf_dict['img_path']
    feat_path=conf_dict['feat_path']
    seq.features.extract_features(img_path,feat_path,'seq')

if __name__ == "__main__":
    conf_path="conf/dane3.cfg"
    make_seq_files(conf_path, new_features=False)