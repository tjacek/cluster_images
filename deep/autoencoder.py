import timeit
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import imp
utils =imp.load_source("utils","/home/user/df/deep_frames/utils.py")

class AutoEncoderReduction(object):
    def __init__(self,path):
        self.autoencoder=utils.read_object(path)

    def transform(self,images):
        x=self.autoencoder.x
        eq=self.autoencoder.get_hidden_values(x)
        dim_reduction = theano.function([x],eq)
        projected=map(dim_reduction,images)
        return projected

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
        self.numpy_rng = np.random.RandomState(123)
        if not theano_rng:
            theano_rng=RandomStreams(self.numpy_rng.randint(2 ** 30))
        self.theano_rng = theano_rng

    def init_hidden_layer(self,W,bhid):
        if not W:
            n_units=self.n_hidden + self.n_visible
            initial_W = np.asarray(
                self.numpy_rng.uniform(
                    low=-4 * np.sqrt(6. / n_units),
                    high=4 * np.sqrt(6. / n_units),
                    size=(self.n_visible, self.n_hidden)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=initial_W, name='W', borrow=True)
        self.W=W
        if not bhid:
            bhid = theano.shared(
                value=np.zeros(
                    self.n_hidden,
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )
	self.b = bhid

    def init_visable_layer(self,bvis):
	if not bvis:
            bvis = theano.shared(
                value=np.zeros(
                    self.n_visible,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )
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
    index = T.lscalar()   
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

    start_time = timeit.default_timer()
    data=dataset.get_batches(n_train_batches,1)
    for epoch in xrange(training_epochs):
        c = []
        for batch_index in xrange(n_train_batches):
            if(( batch_index % 100 ==0)):
                print(batch_index)
            c.append(train_da(data[batch_index]))

        print 'Training epoch %d, cost ' % epoch, np.mean(c)

    end_time = timeit.default_timer()
    training_time = (end_time - start_time)
    print("Training time %d ",training_time)
    return da
