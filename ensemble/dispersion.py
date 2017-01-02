import numpy as np 

def gini_index(raw):
    l,f=lorenz_curve(raw)
    print(f)
    print(l)
    return sum([ f_i - l_i
                 for l_i,f_i in zip(l,f)])

def lorenz_curve(y):
    y=sort_vector(y)
    n=len(y)
    def f_i(i):
        return float(i)/float(n)
    def s_i(i):
        return sum(y[:i])
    s_n=s_i(n)
    l=[ s_i(i)/s_n for i in range(1,n+1)]    
    f=[ f_i(i) for i in range(1,n+1)]   
    return l,f

def rmad(f,y):
    s=0.0
    for i,f_i in enumerate(f):
        for j,f_j in enumerate(f):
            s+= f_i*f_j*float(abs(y[i]-y[j]))
    mi=dot_product(f,y)
    return s/mi

def gini_index_fast(in_vector):
    f=sort_vector(in_vector)
    n=len(in_vector)
    y=range(1,n+1)
    f_sum = gini_sum(f,y,n)
    s_sum=[ f[i]*(f_sum[i-1]+  f_sum[i])
            for i in range(1,n)]
    return 1.0- (dot_product(f,s_sum))#/f_sum[n])  

def gini_sum(f,y,n):
    f_sum=[0.0]
    for i in range(n):
        f_sum.append( f_sum[-1] + f[i] )
    return f_sum

def dot_product(f,y):
    return sum([ f_i * y_i 
                  for f_i,y_i in zip(f,y)])
    
def sort_vector(in_vector):
    if(type(input)!=list):
        in_vector=list(in_vector)
    in_vector.sort()
    return in_vector#np.array(in_vector)

if __name__ == "__main__":
    lorentz1=[0.1,0.2,0.3,0.4]
    lorentz2=[0.0,0.0,0.0,1.0]
    lorentz3=[0.25,0.25,0.25,0.25]
    print(gini_index(lorentz2))
    #test=[0.25,0.25,0.25,0.25]
    #sort_vec=sort_vector(test)
    #print(gini_index(sort_vec))