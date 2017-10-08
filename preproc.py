import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import utils.actions.read
import deep.tools
import basic

def extract_feats(in_path,out_path,dataset_format='mhad_dataset'):
    read_actions=utils.actions.read.ReadActions(dataset_format)
    actions=read_actions(in_path)
    extractor=select_extractor(extractor_type='basic')
    basic_actions=[ action_i.transform(extractor,False) for action_i in actions]
 
    for basic_action_i in basic_actions:
        basic_action_i.to_text_file(out_path)

def select_extractor(extractor_type):
    extractor=basic.get_basic_features()
    if(extractor_type=='basic'):
        return basic.get_basic_features()	
    raise Exception("No preproc")

def select_preproc(preproc):
    if(type(preproc)==dict):	
        preproc=conf_dict['preproc']
    if(preproc=='proj'):
        return deep.tools.ImgPreprocProj()
    elif preproc=='time':    
        return deep.tools.ImgPreproc2D()
    elif preproc=='1D':
        return deep.tools.ImgPreproc1D()
    raise Exception("No preproc")

in_path="../../AArtyk2/time/"
out_path="../../AArtyk2/corel/"
extract_feats(in_path,out_path)