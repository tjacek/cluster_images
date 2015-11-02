import deep
import numpy as np
import theano
import theano.tensor as T
import scipy.misc
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

def make_ml_functions(da,learning_rate=0.1,corruption_level=0.0):
    tilde_x = da.get_corrupted_input(da.x, corruption_level)
    y = get_hidden_values(da.model,tilde_x)
    z = get_reconstructed_input(da.model,y)
    cost = deep.get_crossentropy_loss(da.x,y,z)
    params= da.model.get_params()
    cost,updates=deep.comput_updates(cost, params, learning_rate)
    train = theano.function([da.x],cost,updates=updates)
    test = theano.function([da.x],y)
    get_image = theano.function([da.x],z)
    return train,test,get_image

class AutoEncoder(object):
    def __init__(self,x,n_visible=3200,n_hidden=3000):
        self.init_rng()
        self.model=make_ae_model(n_hidden,n_visible,self.numpy_rng)
        self.x = x
        self.train,self.test,self.get_image=make_ml_functions(self)

    def init_rng(self,theano_rng=None):
        self.numpy_rng,self.theano_rng = deep.make_rng(theano_rng)

    def get_corrupted_input(self, x, corruption_level):
        return self.theano_rng.binomial(size=x.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * x
def get_hidden_values(model, x):
    return deep.get_sigmoid(x,model.W,model.b)

def get_reconstructed_input(model,hidden):
    return deep.get_sigmoid(hidden,model.W_prime,model.b_prime)

def learning_autoencoder(dataset,training_epochs=100,
            learning_rate=0.1,batch_size=5):

    x = T.matrix('x')  
    da = AutoEncoder(x)

    #train_da=make_ml_functions(da,learning_rate)

    timer = utils.Timer()
    n_batches=deep.get_number_of_batches(dataset,batch_size)
    data=deep.get_batches(dataset,n_batches,batch_size)

    for epoch in xrange(training_epochs):
        c = []
        for batch_index in xrange(n_batches):
            #if(( batch_index % 100 ==0)):
            #print(batch_index)
            c.append(da.train(data[batch_index]))

        print 'Training epoch %d, cost ' % epoch, np.mean(c)

    timer.stop()
    timer.show()
    return da

def reconstruct_images(dataset,ae,out_path):
    utils.make_dir(out_path)
    data=deep.standarized_images(dataset)
    for i,img in enumerate(data):
        rec_image=ae.get_image(img)
        img2D=np.reshape(rec_image,(80,40))
        img_path=out_path+dataset.get_name(i)
        print(img_path)
        scipy.misc.imsave(img_path,img2D)
    
