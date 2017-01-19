import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import deep.tools as tools
import deep.convnet
import utils.actions

PREPROC_DICT={"time":tools.ImgPreproc2D,
              "proj":tools.ImgPreprocProj}

class FeatSeq(object):
    def __init__(self,conv):
        self.conv=conv

    def __call__(self,action):
        if(type(action)!=utils.actions.Action):
            raise Exception("Wrong action type " + str(type(action)))
        return [self.conv(img_i)  
                for img_i in action.img_seq]

def make_feat_seq(nn_path,prep_type="time"):
    conv=read_convnet(nn_path)
    return FeatSeq(conv)  

def read_convnet(nn_path,prep_type="time"):
    preproc_method=PREPROC_DICT[prep_type]
    preproc=preproc_method()
    return deep.convnet.get_model(preproc,nn_path,compile=False)

def read_actions(cat_path,action_type='cp_dataset'):
    action_reader=utils.actions.ReadActions(action_type)
    actions=action_reader(cat_path)
    return actions

if __name__ == "__main__":
    nn_path='../dataset1/exp1/nn_full'
    full=make_feat_seq(nn_path)