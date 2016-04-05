import numpy as np 
import numpy.random as rnd

def bool_fun(size=100,max_dim=50):
    data=[]
    y=[]
    for i in range(size):
        cat=i%2
        data.append(bool_seq(max_dim,cat))
        y.append(int(cat))
    mask=get_mask(data,size,max_dim)
    X=to_numpy(data,max_dim)
    y=np.array(y,dtype='int64')
    return X,y,mask

def bool_seq(max_dim,cat):
    length=rnd.randint(1,max_dim)
    seq=np.zeros((length,3))
    for i in range(length):
        seq[i][0]=rnd.randint(2)
        seq[i][1]=rnd.randint(2)
        if(cat):
            seq[i][2]=seq[i][0]*seq[i][1]
        else:
            seq[i][2]=(seq[i][0]+seq[i][1])%2
    return seq

def ABC_lang(numb=100,max_size=50):
    words=[gen_word(max_size) for i in range(numb)]
    y=[i%2 for i in range(numb) ]
    for i,y_i in enumerate(y):
        if(y_i==0):
            words[i]=spoil_word(words[i])
    max_dim=3*max_size
    mask=get_mask(words,numb,max_dim)
    words=to_numpy(words,max_dim)
    return words,y,mask

def gen_word(max_size):
    word=[]
    for value in range(3):
        size=rnd.randint(max_size)
        word+=gen_homo(value,size)		
    return np.array(word)

def spoil_word(word,err=10):
    for i in range(err):
        j=rnd.randint(0,len(word))
        word[j]=(word[j]+1)%2 
    return word

def gen_homo(value,size):
    return [value for i in range(size)] 

def get_mask(words,size,max_dim):
    mask=np.zeros((size,max_dim))
    for i,word_i in enumerate(words):
        length=word_i.shape[0]
        mask[i, :length]=1
    return mask

def to_numpy(X,max_dim):
    size=len(X)
    X_full=[]
    for x_i in X:
        z_size=max_dim-x_i.shape[0]
        #print(x_i.shape)
        if(len(x_i.shape)==1):
            x_i=np.reshape(x_i,(x_i.shape[0],1))
        z_i=np.zeros((z_size,x_i.shape[1]))
        #else:
        #    print("OK")
        #    z_i=np.zeros((z_size,1))
        f_i=np.concatenate([x_i,z_i])
        X_full.append(f_i)
    X_full=np.array(X_full)
    #X_full=np.reshape(X_full,(size,max_dim,1))
    print(X_full.shape)
    return X_full

if __name__ == "__main__":
    #words,y=ABC_lang(200)
    X,y=bool_fun(100,50)
    print(y)	