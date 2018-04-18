import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np
import ensemble,ensemble.nn
import utils.actions.read
import utils.paths.dirs
import utils.text
import seq

class LSTMCls(ensemble.nn.RNNCls):
    def __init__(self, rnn,conv_net):
        super(LSTMCls, self).__init__(conv_net,rnn)

    def get_seq(self,action):
        if(type(action)!=utils.actions.Action):
            raise Exception("Non action type " + str(type(action)))
        print(self.conv_net.keys())
        action_name= action.name.split('.')[0]
        action_name= str(action.cat) +'_'+ action_name
        return self.conv_net[action_name].img_seq	

def make_rnn_ensemble(in_path,lstm_path,dataset_format='cp_dataset'):
    all_seqs=read_seqs(in_path,dataset_format)
    print(all_seqs.keys())
    rnns=ensemble.read_ensemble(lstm_path,with_id=True)
    print(rnns.keys())
    clss=[ LSTMCls(all_seqs[key_i],rnns[key_i]) 
                for key_i in all_seqs.keys()]
    return  ensemble.nn.NNEnsemble(clss)           
        
def read_seqs(in_path,dataset_format='cp_dataset'):
    get_number=utils.text.ExtractNumber() 
    paths=utils.paths.dirs.get_files(in_path)
    action_reader=utils.actions.read.ReadActions(dataset_format,img_seq=False,as_dict=True)
    return { get_number(path_i) :action_reader(path_i)
                for path_i in paths}

if __name__ == "__main__":
    in_path="../../AA_konf/old/full_seqs"
    lstm_path="../../AA_konf/old/all_lstm_50/"
    rnn=make_rnn_ensemble(in_path,lstm_path)
    in_path="../../AA_konf/old/full_seqs/nn_0"
    ensemble.nn.apply_nn(in_path,rnn,img_seq=False)