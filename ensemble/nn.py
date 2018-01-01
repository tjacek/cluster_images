import numpy as np
import ensemble

class NNEnsemble(object):
    def __init__(self, deep_nns):
        self.deep_nns = deep_nns
    
    def __call__(self,seqs):
    	return [ self.get_category(seq_i) 
    	            for seq_i in seqs]
    
    def get_category(self,seq_i):
        return np.argmax(self.get_distribution(seq_i))
    
    def get_distribution(self,seq_i):
        dists=[nn_j.get_distribution(seq_i)
                for nn_j in self.deep_nns]
        dists=np.array(dists)
        return np.sum(dists, axis=0)

def read_datasets(in_path,dataset_format='cp_dataset'):
    datasets_paths=os.listdir(str(in_path))
    datasets=[seq_dataset(path_i,masked=True,dataset_format)[1]
                for path_i in datasets_paths]
    names=datasets[0]['names']
    y=datasets[0]['y']
    datasets=[ to_dir(data_i) for data_i in datasets]
    actions={ name_i:[data_i[name_i] for data_i in datasets] 
                for name_i in names}
    return actions,y,names
    
def to_dir(dataset_j):
    names=dataset_i['names']
    def dir_helper(i,name_i):
        action_i={ 'mask':dataset_j['mask'][i],
                   'x':dataset_j['x'][i],
                   'y':datasets['y'][i]}
        return {name_i:action_i}
    return [ dir_helper(i,name_i) 
                for i,name_i in enumerate(names)]
    