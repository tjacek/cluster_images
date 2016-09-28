import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np 
import utils.dirs
import utils.imgs
import deep.reader
import basic.external

class CombinedFeatures(object):
    def __init__(self, extractors):
        self.extractors = extractors

    def __call__(self,img_i):
        feats=[extr_i(img_i) 
    	        for extr_i in self.extractors]
    	feats=np.concatenate(feats)
    	print(feats.shape)
    	return feats

def build_combined(in_path):
    all_paths=utils.dirs.all_files(in_path)
    nn_reader=deep.reader.NNReader()
    extractors=[nn_reader.read(nn_path_i,0.0)
                  for nn_path_i in all_paths]
    return CombinedFeatures(extractors)

def unify_text_features(in_path):
    all_paths=utils.dirs.all_files(in_path)
    dict_features=[basic.external.read_external(path_i)
                        for path_i in all_paths]

if __name__ == "__main__":
    in_path='../dane/nn'
    img_path='../dane/train_last'
    extractor=build_combined(in_path)
    imgset=utils.imgs.make_imgs(img_path,norm=True)
    extractor(imgset[0])
