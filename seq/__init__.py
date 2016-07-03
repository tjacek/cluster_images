import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import utils
import utils.files as files
import numpy as np
import to_dataset
import deep.lstm

if __name__ == "__main__":
    path='../dataset0/seq/'
    dataset=to_dataset.seq_dataset(path)
    new_dataset=to_dataset.masked_dataset(dataset)
    hyper_params=deep.lstm.get_hyper_params(new_dataset)
    lstm_equ,input_vars=deep.lstm.make_LSTM(hyper_params)
    deep.lstm.compile_lstm(lstm_equ,input_vars,hyper_params)