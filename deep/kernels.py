import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import deep.reader

def get_kernels(conv_net):
    model=conv_net.get_model()
    params=model.params
    filter_size=model.hyperparams["filter_size"]
    kernels=select_kernels(params,filter_size)
    print(len(kernels))
    #inspect(params[0])
    #print(params)

def select_kernels(params,filter_size):
	return [param_i 
	          for param_i in params
	            if param_i.shape[-2:]==filter_size]

def inspect(model):
    print(type(model))
    print(dir(model))

if __name__ == "__main__":
    nn_path="../dataset2/conv_nn"
    nn_reader=deep.reader.NNReader()
    conv_net= nn_reader.read(nn_path,0.0)
    get_kernels(conv_net)