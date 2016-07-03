import lasagne
import numpy as np
import theano
import theano.tensor as T

class LstmModel(object):
    def __init__(self,input_vars,loss,updates):
        self.input_vars=input_vars
        vars_list=[input_vars['in_var'],input_vars['target_var'],input_vars['mask_var']]
        self.train = theano.function(vars_list,loss, updates=updates)
        self.loss = theano.function(vars_list, loss)

def make_LSTM(hyper_params):
    n_batch=hyper_params['n_batch']
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
    l_out = lasagne.layers.DenseLayer(
        l_sum, num_units=n_cats, nonlinearity=lasagne.nonlinearities.softmax)
    input_vars=make_input_vars(l_in,l_mask)
    return l_out,input_vars

def compile_lstm(lstm_equ,input_vars,hyper_params):
    prediction = lasagne.layers.get_output(lstm_equ)
    loss = lasagne.objectives.categorical_crossentropy(prediction,input_vars['target_var'])
    loss = loss.mean()
    params = lasagne.layers.get_all_params(lstm_equ, trainable=True)
    updates = lasagne.updates.adagrad(loss,params, hyper_params['learning_rate'])
    return LstmModel(input_vars,loss,updates)
    
def make_input_vars(l_in,l_mask):
    in_var=l_in.input_var
    target_var = T.ivector('targets')
    mask_var=l_mask.input_var
    return {'in_var':in_var,'target_var':target_var,'mask_var':mask_var}

def get_hyper_params(masked_dataset):
    hyper_params=masked_dataset['params']
    hyper_params['n_hidden']=10
    hyper_params['grad_clip']=100
    hyper_params['learning_rate']=0.001
    return hyper_params	