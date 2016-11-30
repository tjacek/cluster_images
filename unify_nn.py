import numpy as np 
import deep.reader as reader
import basic.combine
import cats

def make_feat_files(conf_path1,conf_path2):
    cats.easy_make_seq(conf_path1, new_feat=True)
    cats.easy_make_seq(conf_path2, new_feat=True)

if __name__ == "__main__":
    conf_path1='conf/exp1.cfg'
    conf_path2='conf/exp2.cfg'	
    make_feat_files(conf_path1,conf_path2)
