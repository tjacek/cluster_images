import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import deep.tools as tools
import deep.convnet
import utils.actions.read
import basic.external

PREPROC_DICT={"time":tools.ImgPreproc2D,
              "proj":tools.ImgPreprocProj}

class FeatSeq(object):
    def __init__(self,conv):
        self.conv=conv
        #print(self.conv.hyperparams)

    def __call__(self,action):
        if(type(action)!=utils.actions.Action):
            raise Exception("Wrong action type " + str(type(action)))
        #img_seq=[img_i/255.0 for img_i in action.img_seq]
        return [self.conv(img_i)  
                for img_i in action.img_seq]

def make_feat_seq(nn_path,prep_type="proj",text_feat=False):
    if(text_feat):
        feat_extr= basic.external.read_external(nn_path)
    else:
        feat_extr=read_convnet(nn_path,prep_type)
    return FeatSeq(feat_extr)  

def read_convnet(nn_path,prep_type="proj"):
    preproc_method=PREPROC_DICT[prep_type]
    preproc=preproc_method()
    nn_reader=deep.reader.NNReader(preproc)
    convnet=nn_reader(nn_path,0.0)
    return convnet

def read_actions(cat_path,action_type='cp_dataset'):
    action_reader=utils.actions.read.ReadActions(action_type,norm=True)
    actions=action_reader(cat_path)
    return actions

if __name__ == "__main__":
    nn_path='../dataset1/exp1/nn_full'
    full=make_feat_seq(nn_path)