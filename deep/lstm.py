import lasagne,pickle
import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from lasagne.random import get_rng
from . import Model

class LstmModel(object):
    def __init__(self,hyper_params,params,input_vars,pred,loss,updates):
        self.hyper_params=hyper_params
        self.params=params
        self.input_vars=input_vars
        vars_list=[input_vars['in_var'],input_vars['target_var'],input_vars['mask_var']]
        self.predict= theano.function([input_vars['in_var'],input_vars['mask_var']],pred)
        self.train = theano.function(vars_list,loss, updates=updates)#,optimizer=None)
        self.loss = theano.function(vars_list, loss)#,optimizer=None)

    def get_category(self,x,mask):
        x=np.expand_dims(x,axis=0)
        mask=np.expand_dims(mask,axis=0)
        return np.argmax(self.predict(x,mask))

    def get_model(self):
        return Model(self.hyper_params,self.params)

def make_LSTM(hyper_params):
    print(hyper_params)
    n_batch=None#hyper_params['n_batch']
    max_seq=hyper_params['max_seq']
    seq_dim=hyper_params['seq_dim']
    n_cats=hyper_params['n_cats']
    n_hidden=hyper_params['n_hidden']
    grad_clip = hyper_params['grad_clip']
    l_in = lasagne.layers.InputLayer(shape=(n_batch, max_seq, seq_dim))
    l_mask = lasagne.layers.InputLayer(shape=(n_batch, max_seq))
    l_forward = lasagne.layers.RecurrentLayer(
        l_in, n_hidden, mask_input=l_mask, grad_clipping=grad_clip,
        W_in_to_hid=lasagne.init.HeUniform(),
        W_hid_to_hid=lasagne.init.HeUniform(),
        nonlinearity=lasagne.nonlinearities.tanh)
    l_backward = lasagne.layers.RecurrentLayer(
        l_in, n_hidden, mask_input=l_mask, grad_clipping=grad_clip,
        W_in_to_hid=lasagne.init.HeUniform(),
        W_hid_to_hid=lasagne.init.HeUniform(),
        nonlinearity=lasagne.nonlinearities.tanh, backwards=True)
    l_forward_slice = lasagne.layers.SliceLayer(l_forward, -1, 1)
    l_backward_slice = lasagne.layers.SliceLayer(l_backward, 0, 1)
    l_sum = lasagne.layers.ConcatLayer([l_forward_slice, l_backward_slice])
    l_drop= lasagne.layers.DropoutLayer( lasagne.layers.FlattenLayer(l_sum))
    l_out = lasagne.layers.DenseLayer(
        l_drop, num_units=n_cats, nonlinearity=lasagne.nonlinearities.softmax) 
    input_vars=make_input_vars(l_in,l_mask)
    return l_out,input_vars

def compile_lstm(lstm_equ,input_vars,hyper_params):
    prediction = lasagne.layers.get_output(lstm_equ)
    params = lasagne.layers.get_all_param_values(lstm_equ)
    loss = lasagne.objectives.categorical_crossentropy(prediction,input_vars['target_var'])
    loss = loss.mean()
    params = lasagne.layers.get_all_params(lstm_equ, trainable=True)
    #updates = lasagne.updates.nesterov_momentum(loss,params, hyper_params['learning_rate'])
    updates =lasagne.updates.adagrad(loss,params, hyper_params['learning_rate'])
    return LstmModel(hyper_params,params,input_vars,prediction,loss,updates)

def make_input_vars(l_in,l_mask):
    in_var=l_in.input_var
    target_var = T.ivector('targets')
    mask_var=l_mask.input_var
    return {'in_var':in_var,'target_var':target_var,'mask_var':mask_var}

def get_hyper_params(masked_dataset):
    hyper_params=masked_dataset['params']
    hyper_params['n_hidden']=100
    hyper_params['grad_clip']=100
    hyper_params['learning_rate']=0.001
    hyper_params['learning_rate']=0.001
    hyper_params['momentum']=0.9
    return hyper_params	