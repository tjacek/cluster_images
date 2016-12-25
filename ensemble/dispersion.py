import numpy as np 

def gini_index(in_vector):
    in_vector=sort_vector(in_vector)
    print(in_vector)
    n=float(len(in_vector))
    gini_value=sum([(2.0*i - n - 1.0)*y_i
                    for i,y_i in enumerate(in_vector)])  
    gini_value/= n * sum(in_vector)
    return gini_value

def gini_sum(f,y):
    s_array=sum([ f_i * y_i 
                  for f_i,y_i in zip(f,y)])
    return s_array
    
def sort_vector(in_vector):
    if(type(input)!=list):
        in_vector=list(in_vector)
    in_vector.sort()
    return in_vector#np.array(in_vector)

if __name__ == "__main__":
    test=[4.0,1.0,7.0,5.0,3.0,2.0]
    sort_vec=sort_vector(test)
    print(gini_index(sort_vec))