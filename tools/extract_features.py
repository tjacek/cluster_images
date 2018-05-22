import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
#import preproc
import utils.paths
import deep.convnet,deep.tools
#import seq.features,seq.dtw_feat
import utils.actions.read

class MakeFeats(object):
    def __init__(self,preproc_type='time',extractor_type='deep',
                        dataset_format='cp_dataset'):
        self.dataset_format=dataset_format
        self.preproc_type=preproc_type
        self.extractor_type=extractor_type

    def __call__(self,img_path,out_path,feat_path=None):
        #if(self.extractor=='deep'):
        #    extractor_desc=('deep',feat_path)
        #select_extractor(self.extractor_type,self.preproc_type)
        preproc=deep.tools.ImgPreproc2D()
        extractor=deep.convnet.get_model(10,preproc,feat_path,compile=True,model_p=0.5)

        read_actions=utils.actions.read.ReadActions(dataset_format)
        actions=read_actions(in_path)
        new_actions=[ action_i.transform(extractor,False) for action_i in actions] 
        save_actions=utils.actions.read.SaveActions(img_actions=False)
        save_actions(out_path)

if __name__ == "__main__":
    img_path="../../Documents/DD/cats"
    out_path="../../Documents/DD/seqs"
    feat_path="../../Documents/DD/nn"
    make_feats=MakeFeats()
    make_feats(img_path,out_path,feat_path)
#    out_path=None#"../../methods/Vb/o"  #"../../konf3/max_z/simple.txt"
#    make_feats(img_path,feat_path,out_path=None,dataset_format='cp_dataset')
