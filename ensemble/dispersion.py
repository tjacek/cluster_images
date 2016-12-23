import numpy as np 

def sort_vector(in_vector):
    if(type(input)!=list):
        in_vector=list(in_vector)
    in_vector.sort()
    return np.array(in_vector)

if __name__ == "__main__":
    test=[4.0,1.0,7.0,5.0,3.0,2.0]
    sort_vec=sort_vector(test)
    print(sort_vec)