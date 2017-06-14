import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import utils.selection
import utils.actions

if __name__ == "__main__":
    in_path='../../exper/unified/full'
    out_path='../../exper/unified/train'
    selector=utils.selection.SelectModulo(1)
    utils.actions.apply_select(in_path,out_path, selector,dataset_format='cp_dataset')