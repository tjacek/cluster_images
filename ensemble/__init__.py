import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import deep.reader,deep.tools as tools
import seq.lstm
import utils.paths

def lstm_ensemble(in_path,out_path):
    seq_paths=os.listdir(str(in_path))
    #out_path=utils.paths.Path(out_path)
    for i,seq_i in enumerate(seq_paths):
        seq_i=utils.paths.Path(in_path+'/'+seq_i)
        lstm_i=out_path +'/'+ ('lstm_'+str(i))
        seq.lstm.create_dataset(seq_i,lstm_i)
 
def read_ensemble(nn_dir):
    preproc=tools.ImgPreproc2D()
    nn_reader=deep.reader.NNReader(preproc)
    nn_paths=os.listdir(str(nn_dir))
    nn_paths=[str(nn_dir)+str(nn_path_i)
                for nn_path_i in nn_paths]    
    return [ nn_reader(nn_i) 
                for nn_i in nn_paths]

if __name__ == "__main__":
    in_path="../../AArtyk/all_seqs"
    out_path="../../AArtyk_exp/lstm"
    lstm_ensemble(in_path,out_path)