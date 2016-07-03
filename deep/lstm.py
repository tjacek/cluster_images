import lasagne
import numpy as np
import theano
import theano.tensor as T

def make_LSTM(hyper_params):
    n_batch=hyper_params['n_batch']
    max_seq=hyper_params['max_seq']
    seq_dim=hyper_params['seq_dim']
    n_hidden=10
    grad_clip = 100
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
        l_sum, num_units=1, nonlinearity=lasagne.nonlinearities.tanh)
    return l_out

#def get_hyper_params(masked_dataset):
	