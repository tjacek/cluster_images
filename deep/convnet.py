import numpy as np
import theano
import theano.tensor as T
import lasagne


def build_convnet(params,n_cats):
	input_shape=(None, 1, params["dimX"], params["dimY"]),
	n_filters=params["num_filters"]
	fliter_size2D=(params["filter_size"],params["filter_size"])
	pool_size2D=(params["pool_size"],params["pool_size"])
    p_drop=params["p"]
	in_layer = lasagne.layers.InputLayer(
		       shape=input_shape
               input_var=input_var)
	conv_layer1 = lasagne.layers.Conv2DLayer(
            in_layer, num_filters=n_filters, filter_size=filter_size2D,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
	pool_layer1 = lasagne.layers.MaxPool2DLayer(conv_layer1, pool_size=pool_size2D)
    conv_layer2 = lasagne.layers.Conv2DLayer(
            pool_layer1, num_filters=num_filters, filter_size=filter_size2D,
            nonlinearity=lasagne.nonlinearities.rectify)
    pool_layer2 = lasagne.layers.MaxPool2DLayer(conv_layer2, pool_size=pool_size2D)
    dropout = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(pool_layer2, p=p_drop),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)
    out_layer = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(dropout, p=p_drop),
            num_units=n_cats,
            nonlinearity=lasagne.nonlinearities.softmax)

def default_params():
    return {"dimX":60,"dimY":60,"num_filters":32,
              "filter_size":5,"pool_size":2,"p":0.5}
