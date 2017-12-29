import deep.reader,deep.tools as tools

def read_ensemble(nn_dir):
    preproc=tools.ImgPreproc2D()
    nn_reader=deep.reader.NNReader(preproc)
    nn_paths=os.listdir(str(nn_dir))
    nn_paths=[str(nn_dir)+str(nn_path_i)
                for nn_path_i in nn_paths]    
    for nn_i in nn_paths:
        print(nn_i)
    return [ nn_reader(nn_i) 
                for nn_i in nn_paths]