import lasagne
import numpy as np
import theano
import theano.tensor as T
import utils.files as files
import deep.tools as tools

class StackedAE(object):
    def __init__(self,autoencoder,n_cats):
        self.target_var = T.ivector('targets')
    	W_hid,b_hid,W_out,b_out=autoencoder.get_numpy()
    	input_size,hidden_size=W_hid.shape
    	self.l_in =  lasagne.layers.InputLayer(shape=(None,input_size),W=W_hid,b=b_hid)#,input_var=self.input_var)
        self.l_hid = lasagne.layers.DenseLayer(self.l_in, num_units=hidden_size)
        softmax = lasagne.nonlinearities.softmax
        self.network = lasagne.layers.DenseLayer(self.l_hid,n_cats, nonlinearity=softmax)
        self.get_loss()
        self.prob = theano.function([self.get_input_var()], self.prediction)

    def get_robust_category(self,img,threshold=0.7):
        dist=self.prob(img)
        if(np.amax(dist)<threshold):
            return 0
        return tools.dist_to_category(dist) + 1

    def get_category(self,img):
        dist=self.prob(img)
        return tools.dist_to_category(dist)

    def get_loss(self):
        self.prediction = lasagne.layers.get_output(self.network)
        loss = lasagne.objectives.categorical_crossentropy(self.prediction, self.target_var)
        self.loss = loss.mean()

    def get_params(self):
        return lasagne.layers.get_all_params(self.network, trainable=True)

    def get_updates(self):
        params=self.get_params()
        return lasagne.updates.nesterov_momentum(
                #self.prediction,
                self.loss,
                params, learning_rate=0.01, momentum=0.9)

    def get_input_var(self):
        return self.l_in.input_var