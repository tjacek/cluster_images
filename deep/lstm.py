import theano
import theano.tensor as T
import lasagne
import numpy as np
import tools
import gen

class RNN(object):
    def __init__(self,N_BATCH, MAX_LENGTH,dim_of_vector=1,N_HIDDEN=1,GRAD_CLIP = 100):
        
        self.l_in = lasagne.layers.InputLayer(shape=(N_BATCH, MAX_LENGTH, dim_of_vector))
        self.l_mask = lasagne.layers.InputLayer(shape=(N_BATCH, MAX_LENGTH))
        self.l_forward = lasagne.layers.RecurrentLayer(
            self.l_in, N_HIDDEN, mask_input=self.l_mask, 
            grad_clipping=GRAD_CLIP,
            W_in_to_hid=lasagne.init.HeUniform(),
            W_hid_to_hid=lasagne.init.HeUniform(),
            nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True)
        self.l_backward = lasagne.layers.RecurrentLayer(
            self.l_in, N_HIDDEN, mask_input=self.l_mask, grad_clipping=GRAD_CLIP,
            W_in_to_hid=lasagne.init.HeUniform(),
            W_hid_to_hid=lasagne.init.HeUniform(),
            nonlinearity=lasagne.nonlinearities.tanh,
            only_return_final=True, backwards=True)
        self.l_concat = lasagne.layers.ConcatLayer([self.l_forward, self.l_backward])
        self.l_out = lasagne.layers.DenseLayer(
               self.l_concat, num_units=1, nonlinearity=lasagne.nonlinearities.tanh)

        self.target_values = T.vector('target_output')
        network_output = lasagne.layers.get_output(self.l_out)
        self.predicted_values = network_output.flatten()
        #self.cost = T.mean((self.predicted_values - self.target_values)**2)
        self.cost = T.mean(lasagne.objectives.binary_crossentropy(self.predicted_values,self.target_values))
        self.apply=theano.function([self.get_input_var(),self.l_mask.input_var],self.predicted_values)

    def get_input_var(self):
        return self.l_in.input_var

    def get_params(self):
        return lasagne.layers.get_all_params(self.l_out, trainable=True)

    def get_updates(self):
        all_params=self.get_params()
        LEARNING_RATE= .001
        return lasagne.updates.adagrad(self.cost, all_params, LEARNING_RATE)


def train_seq(X,y,mask,model,iters=100):
    input_var=model.get_input_var()
    updates=model.get_updates()
    train = theano.function([input_var, model.target_values, 
                            model.l_mask.input_var],
                            model.cost, updates=updates)
    x_batch,n_batch=tools.get_batch(X)
    y_batch,n_batch=tools.get_batch(y)
    mask_batch,n_batch=tools.get_batch(mask)
    print(n_batch)
    for epoch in range(iters):
        cost_e = []
        for i in range(n_batch):
            x_i=x_batch[i]
            y_i=y_batch[i]
            mask_i=mask_batch[i]
            #print(x_i.shape)
            #print(y_i.shape)
            #print(mask_i.shape)
            loss_i=train(x_i,y_i,mask_i)
            cost_e.append(loss_i)
            cost_mean=np.mean(cost_e)
            #print(model.apply(x_i,mask_i))
        print(str(epoch) + " "+str(cost_mean))
    return model

if __name__ == "__main__":
    X,y,mask=gen.bool_fun(size=199)#gen.ABC_lang(299)
    print(y)
    print(X.shape)
    rnn=RNN(10,X.shape[1],dim_of_vector=X.shape[2])
    train_seq(X,y,mask,rnn) 
    #tools.make_simple_layer(100,20,postfix="f") 