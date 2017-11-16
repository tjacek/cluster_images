import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import utils.paths.dirs
import utils.actions.read
import deep.tools,deep.reader
import basic
import seq.dtw_feat
import feat.global_feat
import utils.paths.dirs 
import gc

def all_feats(in_path,seq_path,out_path,aggregate_type='dtw',
              extractor_type='deep',dataset_format='mhad_dataset'):
    local_feats(in_path,seq_path,extractor_type,dataset_format)
    #print("**********")
    #print(str(seq_path))
    gc.collect()
    global_feats(seq_path,out_path,aggregate_type,dataset_format)
    #gc.collect()

def global_feats(seq_path,out_path,aggregate_type='dtw',dataset_format='mhad_dataset'):
    if(aggregate_type=='dtw'):
        seq.dtw_feat.make_dtw_feat(seq_path,out_path,dataset_format=dataset_format)
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

@utils.paths.path_args
def all_models(in_path,seq_path,out_path,nn_path):
    nn_paths= utils.paths.dirs.get_files(nn_path,False)
    utils.paths.dirs.make_dir(seq_path)
    #seq_paths=[ seq_path.replace(in_i)  for in_i in nn_paths]
    utils.paths.dirs.make_dir(out_path)
    #out_paths=[ out_path.replace(in_i)  for in_i in nn_paths]
    for nn_path_i in nn_paths:#in enumerate(nn_paths):
        #print(str(seq_paths[i]))
        seq_path_i= seq_path.replace(nn_path_i)
        out_path_i= out_path.replace(nn_path_i)
        all_feats(in_path,seq_path_i,out_path_i,#,str(seq_path[i]),out_path[i],
            aggregate_type='simple',extractor_type=get_deep(nn_path_i))

#img_path='../../AArtyk2/time'
#nn_path="../../AArtyk2/deep/all/nn_all"
#seq_path="../../AArtyk2/deep/all/seq"
#out_path="../../AArtyk2/deep/all/simple.txt"
in_path="../../AArtyk2/time"
nn_path="../../AArtyk3/models"
seq_path="../../AArtyk3/seqs"
out_path="../../AArtyk3/feats"
all_models(in_path,seq_path,out_path,nn_path)
#all_feats(in_path,seq_path,out_path+'/a1.txt',aggregate_type='simple',extractor_type=get_deep(nn_path+'/nn_1'))
