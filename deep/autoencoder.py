import deep
import numpy as np
import theano
import theano.tensor as T
import imp
utils =imp.load_source("utils","/home/user/df/deep_frames/utils.py")

class AutoencoderModel(object):
    def __init__(self,W,b,b_prime):
        self.W=W
        self.b=b
        self.b_prime=b_prime
        self.W_prime = W.T

    def get_params(self):
        return [self.W, self.b, self.b_prime]

def make_ae_model(n_hidden,n_visible,numpy_rng):
    initial_W =deep.init_random(n_hidden,n_visible,numpy_rng)
    W = deep.make_var(initial_W,'W')
    init_b=deep.init_zeros(n_hidden)
    bhid = deep.make_var(init_b,'b')
    init_bvis=deep.init_zeros(n_visible)
    bvis = deep.make_var(init_bvis,"bvis")
    return AutoencoderModel(W,bhid,bvis)

def make_ml_functions(da,learning_rate):
    cost, updates = da.get_cost_updates(
        corruption_level=0.0,
        learning_rate=learning_rate
    )
    train_da = theano.function([da.x],cost,updates=updates)
    return train_da

class AutoEncoder(object):
    def __init__(self,x,n_visible=3200,n_hidden=800):
        self.init_rng()
        self.model=make_ae_model(n_hidden,n_visible,self.numpy_rng)
        self.x = x

    def init_rng(self,theano_rng=None):
        self.numpy_rng,self.theano_rng = deep.make_rng(theano_rng)

    def get_corrupted_input(self, input, corruption_level):
        return self.theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * input

    def get_hidden_values(self, x):
        return deep.get_sigmoid(x,self.model.W,self.model.b)

    def get_reconstructed_input(self, hidden):
        return deep.get_sigmoid(hidden,self.model.W_prime,self.model.b_prime)

    def get_cost_updates(self, corruption_level, learning_rate):
        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        cost = deep.get_crossentropy_loss(self.x,y,z)
        params= self.model.get_params()
        return deep.comput_updates(cost, params, learning_rate)


def learning_autoencoder(dataset,training_epochs=100,
            learning_rate=0.1,batch_size=5):
    #n_train_batches=len(dataset)

    x = T.matrix('x')  
    da = AutoEncoder(x)

    train_da=make_ml_functions(da,learning_rate)

    timer = utils.Timer()
    n_batches=deep.get_number_of_batches(dataset,batch_size)
    data=deep.get_batches(dataset,n_batches,batch_size)

    for epoch in xrange(training_epochs):
        c = []
        for batch_index in xrange(n_batches):
            #if(( batch_index % 100 ==0)):
            print(batch_index)
            c.append(train_da(data[batch_index]))

        print 'Training epoch %d, cost ' % epoch, np.mean(c)

    timer.stop()
    timer.show()
    return da
