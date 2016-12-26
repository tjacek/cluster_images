import numpy as np 

def gini_index(in_vector):
    f=sort_vector(in_vector)
    n=len(in_vector)
    y=range(1,n+1)
    f_sum = gini_sum(f,n)
    s_sum=[f_sum[i-1]+  f_sum[i]
            for i in range(1,n+1)]
    return 1.0-dot_product(f,s_sum)   

def gini_sum(f,n):
    f_sum=[0]
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
    test=[0.25,0.25,0.25,0.25]
    sort_vec=sort_vector(test)
    print(gini_index(sort_vec))