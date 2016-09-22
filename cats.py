import preproc
import seq.features
import utils.conf
    
if __name__ == "__main__":
    conf_path="conf/dataset0.cfg"
    conf_dict=utils.conf.read_config(conf_path)
    print(conf_dict)
    preproc.make_features(conf_dict)
    img_path=conf_dict['img_path']
    feat_path=conf_dict['feat_path']
    seq.features.extract_features(img_path,feat_path,'seq')