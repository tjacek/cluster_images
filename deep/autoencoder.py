import deep
import numpy as np
import theano
import theano.tensor as T
import imp
utils =imp.load_source("utils","/home/user/df/deep_frames/utils.py")

class AutoEncoder(object):
    def __init__(
        self,theano_rng=None,input=None,n_visible=3200,
        n_hidden=800,W=None,bhid=None,bvis=None):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.init_rng(theano_rng)
        self.init_hidden_layer(W,bhid)
        self.init_visable_layer(bvis)
        self.init_input(input)
        self.params = [self.W, self.b, self.b_prime]

    def init_rng(self,theano_rng):
        self.numpy_rng,self.theano_rng = deep.make_rng(theano_rng)

    def init_hidden_layer(self,W,bhid):
        if not W:
            initial_W =deep.init_random(self.n_hidden,self.n_visible,self.numpy_rng)
            W = deep.make_var(initial_W,'W')
        self.W=W
        if not bhid:
            init_b=deep.init_zeros(self.n_hidden)
            bhid = deep.make_var(init_b,'b')
	self.b = bhid

    def init_visable_layer(self,bvis):
	if not bvis:
            init_bvis=deep.init_zeros(self.n_visible)
            bvis = deep.make_var(init_bvis,"bvis")
        self.b_prime = bvis
        self.W_prime = self.W.T

    def init_input(self,input):
        if input is None:
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

    def get_corrupted_input(self, input, corruption_level):
        return self.theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * input

    def get_hidden_values(self, input):
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost_updates(self, corruption_level, learning_rate):
        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        cost = T.mean(L)
        gparams = T.grad(cost, self.params)
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]
        return (cost, updates)

def learning_autoencoder(dataset,training_epochs=15,
            learning_rate=0.1,batch_size=25):
    n_train_batches=len(dataset)
    x = T.matrix('x')  

    da = AutoEncoder(input=x)

    cost, updates = da.get_cost_updates(
        corruption_level=0.,
        learning_rate=learning_rate
    )

    train_da = theano.function(
        [x],
        cost,
        updates=updates
    )

    timer = utils.Timer()
    data=dataset.get_batches(n_train_batches,1)
    for epoch in xrange(training_epochs):
        c = []
        for batch_index in xrange(n_train_batches):
            if(( batch_index % 100 ==0)):
                print(batch_index)
            c.append(train_da(data[batch_index]))

        print 'Training epoch %d, cost ' % epoch, np.mean(c)

    timer.stop()
    timer.show()
    return da
