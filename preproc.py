import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import utils.paths.dirs
import utils.actions.read
import deep.tools,deep.reader
import basic
import seq.dtw_feats
import feat.global_feat
import utils.paths.dirs 
import gc

def deep_feats(in_path,seq_path,out_path,aggregate_type='simple',
              extractor_type='deep',dataset_format='mhad_dataset'):
    seq_path=prepare_path(seq_path)
    out_path=prepare_path(out_path)
    if(os.path.isdir(str(extractor_type))):
        nn_paths=utils.paths.dirs.get_files(extractor_type,dirs=False)
        seq_paths=[ seq_path.replace(nn_i) 
                    for nn_i in nn_paths]
        out_paths=[ out_path.replace(nn_i) 
                    for nn_i in nn_paths]
        for i,nn_i in enumerate(nn_paths):
            extractor_type=get_deep(nn_i)
            all_feats(in_path,seq_paths[i],out_paths[i],
                aggregate_type,extractor_type,dataset_format)
    else:
        all_feats(in_path,seq_path,out_path,
            aggregate_type,extractor_type,dataset_format)

def prepare_path(path):
    os.mkdir(str(path))
    return utils.paths.Path(path)

def all_feats(in_path,seq_path,out_path,aggregate_type='dtw',
              extractor_type='deep',dataset_format='mhad_dataset'):
    local_feats(in_path,seq_path,extractor_type,dataset_format)
    gc.collect()
    global_feats(seq_path,out_path,aggregate_type,dataset_format)

def global_feats(seq_path,out_path,aggregate_type='dtw',dataset_format='mhad_dataset'):
    if(aggregate_type=='dtw'):
        seq.dtw_feats.make_dtw_feat(seq_path,out_path,dataset_format=dataset_format)
    else:
        feat.global_feat.get_global_features(seq_path,out_path,dataset_format=dataset_format)

def local_feats(in_path,seq_path,extractor_type='deep',dataset_format='mhad_dataset'):
    read_actions=utils.actions.read.ReadActions(dataset_format)
    actions=read_actions(in_path)
    extractor=select_extractor(extractor_type)
    basic_actions=[ action_i.transform(extractor,False) for action_i in actions] 
    utils.paths.dirs.make_dir(seq_path)
    for basic_action_i in basic_actions:
        print(out_path)
        basic_action_i.to_text_file(seq_path)

def select_extractor(extractor_type,preproc_type='time'):
    extractor_id,extractor_desc=decompose(extractor_type)
    if(extractor_id=='basic'):
        return basic.get_basic_features()
    elif(extractor_id=='deep'):
        preproc=select_preproc(preproc_type)
        return get_deep_reader(extractor_desc,preproc)
    raise Exception("No extractor")

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

def decompose(extractor_type):
    extractor_desc=extractor_type
    if(type(extractor_type)==dict):
        extractor_id=extractor_type['extractor_id']
    else:
        extractor_id=extractor_type
    return extractor_id,extractor_desc

def get_deep(nn_path):
    return {'extractor_id':'deep','nn_path':nn_path}

in_path="../../AArtyk2/time"
nn_path="../../AArtyk3/all_models"
seq_path="../../AArtyk3/all_seqs"
out_path="../../AArtyk3/all_feats"
deep_feats(in_path,seq_path,out_path,aggregate_type='simple',
            extractor_type=nn_path,dataset_format='mhad_dataset')
#all_models(in_path,seq_path,out_path,nn_path)
#all_feats(in_path,seq_path,out_path+'/a1.txt',aggregate_type='simple',extractor_type=get_deep(nn_path+'/nn_1'))
