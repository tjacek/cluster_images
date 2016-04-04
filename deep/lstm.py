import theano
import theano.tensor as T
import lasagne
import numpy as np
import tools

class LSTM(object):
    def __init__(self,N_HIDDEN=10,input_dim=1):
        self.l_in=lasagne.layers.InputLayer(shape=(None,None,input_dim))
        l_mask=lasagne.layers.InputLayer(shape=(None,None))
        gate_parameters=lasagne.layers.recurrent.Gate(W_in=lasagne.init.Orthogonal(),
                                            W_hid=lasagne.init.Orthogonal(),b=lasagne.init.Constant(0.))
        cell_parameters=lasagne.layers.recurrent.Gate(W_in=lasagne.init.Orthogonal(),
                                              W_hid=lasagne.init.Orthogonal(),W_cell=None,b=lasagne.init.Constant(0.),
                                              nonlinearity=lasagne.nonlinearities.tanh)
        self.l_lstm=lasagne.layers.recurrent.LSTMLayer(self.l_in,N_HIDDEN,mask_input=l_mask,
                                        ingate=gate_parameters,forgetgate=gate_parameters,
                                          cell=cell_parameters,outgate=gate_parameters,
                                          learn_init=True,grad_clipping=100.)
        self.prediction_symb = lasagne.layers.get_output(self.l_lstm)
        self.get_loss()
        self.prediction=theano.function([self.get_input_var()], self.prediction_symb)

    def get_input_var(self):
        return self.l_in.input_var

    def get_loss(self):
        target_var=self.get_input_var()
        self.loss = lasagne.objectives.squared_error(self.prediction_symb,target_var)
        self.loss = self.loss.mean()



if __name__ == "__main__":
    #words,y=ABC_lang(200)
    #print(words)
    #LSTM() 
    tools.make_simple_layer(100,20,postfix="f") 