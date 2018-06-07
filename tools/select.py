import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import utils.selection
import utils.actions
import utils.actions.tools

if __name__ == "__main__":
    in_path='../../Documents/AC1/full'
    out_path='../../Documents/AC1/train'
    select_actions=utils.actions.tools.ActionSelection(in_seq=True,out_seq=True, dataset_format='cp_dataset')
    select_actions(in_path,out_path, selector=1)