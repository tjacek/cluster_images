import numpy as np 
import numpy.random as rnd

def ABC_lang(numb=100,max_size=50):
    words=[gen_word(max_size) for i in range(numb)]
    y=[i%2 for i in range(numb) ]
    for i,y_i in enumerate(y):
        if(y_i==0):
            words[i]=spoil_word(words[i])
    #X=np.array(words)
    return words,y

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

if __name__ == "__main__":
    words,y=ABC_lang(200)
    print(words)	