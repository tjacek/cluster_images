import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import deep.tools as tools
import deep.convnet

PREPROC_DICT={"time":tools.ImgPreproc2D,
              "proj":tools.ImgPreprocProj}

def read_convnet(nn_path,prep_type="time"):
    preproc_method=PREPROC_DICT[prep_type]
    preproc=preproc_method()
    return deep.convnet.get_model(preproc,nn_path,compile=False)

if __name__ == "__main__":
    nn_path='../dataset1/exp1/nn_17'
    read_convnet(nn_path,"time")