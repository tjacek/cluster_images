import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import utils
import utils.files as files
import numpy as np
import to_dataset

if __name__ == "__main__":
    path='../dataset0/seq/'
    dataset=to_dataset.seq_dataset(path)
    new_dataset=to_dataset.masked_dataset(dataset)
    print(new_dataset['params'])