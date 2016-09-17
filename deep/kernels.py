import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np
import cv2
import deep.reader
import utils.dirs

class ApplyKernels(object):
    def ApplyKernels(kernels):
        self.curry_kern=[CurryKernel(kern_i)
                for kern_i in kernels]
    
    def __call__(self,img_i):
        return [kern_i(img_i) 
                for kern_i in self.curry_kern]

class CurryKernel(object):
    def __init__(self, kernel):
        self.kernel=kernel

    def __call__(self,img):
        return cv2.filter2D(img,self.kernel)

def get_kernels(conv_net):
    model=conv_net.get_model()
    params=model.params
    filter_size=model.hyperparams["filter_size"]
    kernels=select_kernels(params,filter_size)
    kernels=flatten_kernels(kernels[0])
    show_shapes(kernels)
    save_kernels(kernels,'kernels')

def select_kernels(params,filter_size):
	return [param_i 
	          for param_i in params
	            if param_i.shape[-2:]==filter_size]

def flatten_kernels(kernel_numpy):
    return [np.reshape(kernel_i, (-1,5)) 
            for kernel_i in kernel_numpy]

def inspect(model):
    print(type(model))
    print(dir(model))

def show_shapes(arrays):
    shapes=[array_i.shape for array_i in arrays]
    print(shapes)

def save_kernels(kernels,out_path):
    utils.dirs.make_dir(out_path)
    paths=[out_path +'/cls'+str(i)+'.jpg'
            for i,kern_i in enumerate(kernels)]
    for kern_i,path_i in zip(kernels,paths):
        kern_i*=200
        cv2.imwrite(path_i,kern_i)

if __name__ == "__main__":
    nn_path="../dataset2/conv_nn"
    nn_reader=deep.reader.NNReader()
    conv_net= nn_reader.read(nn_path,0.0)
    get_kernels(conv_net)