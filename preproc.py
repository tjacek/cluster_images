import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import utils.paths.dirs
import utils.actions.read
import deep.tools,deep.reader
import basic
import seq.dtw_feat
import feat.global_feat

def global_feats(seq_path,out_path,aggregate_type='dtw',dataset_format='mhad_dataset'):
    if(aggregate_type=='dtw'):
        seq.dtw_feat.make_dtw_feat(seq_path,out_path,dataset_format=dataset_format)
    else:
        feat.global_feat.get_global_features(seq_path,out_path,dataset_format=dataset_format)

def local_feats(in_path,out_path,extractor_type='deep',dataset_format='mhad_dataset'):
    read_actions=utils.actions.read.ReadActions(dataset_format)
    actions=read_actions(in_path)
    extractor=select_extractor(extractor_type)
    basic_actions=[ action_i.transform(extractor,False) for action_i in actions]
    utils.paths.dirs.make_dir(out_path)
    for basic_action_i in basic_actions:
        basic_action_i.to_text_file(out_path)

def select_extractor(extractor_type,preproc_type='time'):
    extractor_id,extractor_desc=decompose(extractor_type)
    if(extractor_id=='basic'):
        return basic.get_basic_features()
    elif(extractor_id=='deep'):
        preproc=select_preproc(preproc_type)
        return get_deep_reader(extractor_desc,preproc)
    raise Exception("No extractor")

def decompose(extractor_type):
    extractor_desc=extractor_type
    if(type(extractor_type)==dict):
        extractor_id=extractor_type['extractor_id']
    else:
        extractor_id=extractor_type
    return extractor_id,extractor_desc

def select_preproc(preproc_type):
    if(type(preproc_type)==dict):	
        preproc=conf_dict['preproc']
    if(preproc_type=='proj'):
        return deep.tools.ImgPreprocProj()
    elif preproc_type=='time':    
        return deep.tools.ImgPreproc2D()
    elif preproc_type=='1D':
        return deep.tools.ImgPreproc1D()
    raise Exception("No preproc")

def get_deep_reader(nn_path,preproc):
    if(type(nn_path)==dict):  
        nn_path=nn_path['nn_path']
    nn_reader=deep.reader.NNReader(preproc)
    extractor=nn_reader(nn_path)
    return extractor

def get_deep(nn_path):
    return {'extractor_id':'deep','nn_path':nn_path}

img_path='../../AArtyk2/time'
nn_path="../../AArtyk2/deep/16/nn_16"
seq_path="../../AArtyk2/deep/16/seq"
out_path="../../AArtyk2/basic/simple/simple.txt"
#global_feats(seq_path,out_path,'basic',dataset_format='mhad_dataset')

local_feats(img_path,seq_path,get_deep(nn_path))