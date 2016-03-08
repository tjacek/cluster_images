import cv2
import numpy as np

def get_shape_context(in_path):
    print(in_path)
    points,img=read_points(in_path)
    if(len(points)==0):
        return None	
    extr_points=extrem_points(points)
    #center=center_of_image(img)#center_of_mass(points)
    hist=get_multiple_histograms(extr_points,points)
    return hist.flatten()

def read_points(in_path):
    img=cv2.imread(in_path,cv2.IMREAD_GRAYSCALE)
    points=[]
    for (x,y), value in np.ndenumerate(img):
      if(value!=0):
          points.append(make_point(x,y))
    return points,img

def extrem_points(points):
    points=np.array(points)
    max_id=[np.argmax(points[:,0]),np.argmax(points[:,1])]
    min_id=[np.argmin(points[:,0]),np.argmin(points[:,1])]
    indices=max_id+min_id
    extrem_points=[points[p_id] for p_id in indices ]
    print(extrem_points)
    return extrem_points

def center_of_mass(points):
    np_points=np.array(points)
    size=float(len(points))
    center=np_points.sum(axis=0)
    center/=size
    return center

def center_of_image(img):
    x=float(img.shape[0])/2.0
    y=float(img.shape[1])/2.0
    return make_point(x,y)

def normalize_dist(dists,vectors):
    avg_dist=sum(dists)/float(len(dists))
    dists=np.array(dists)
    dists/=avg_dist
    for vector_i in vectors:
        vector_i/=avg_dist
    return dists,vectors

def get_multiple_histograms(key_points,points):
    hists=[get_single_histogram(key_point_i,points)  
                 for key_point_i in key_points]
    mult_hist=[]
    for hist_i in hists:
        mult_hist+=list(hist_i)
    return np.array(mult_hist)

def get_single_histogram(center,points):
    vectors=get_vectors(center,points)
    dists=get_distances(vectors)
    dists,vectors=normalize_dist(dists,vectors)
    return get_histogram(dists,vectors)

def get_histogram(dists,vectors,dist_bins=6,theta_bins=18):
    thetas=[ vec_i[0]/dist_i for vec_i,dist_i in zip(vectors,dists)]
    hist=np.zeros((dist_bins+1,theta_bins+1))
    for dist_i,theta_i in zip(dists,thetas):
        y_i=(theta_i+1.0)/2.0
        y_i=np.floor(theta_bins*y_i)
        x_i=compute_tan_bin(dist_i,dist_bins)
        if(x_i>=dist_bins):
            x_i= -1#log_bins
        if(y_i>=theta_bins):
            y_i= -1#theta_bins	
        if(0<=x_i and 0<=y_i):
            hist[x_i][y_i]+=1.0
    hist/=float(len(dists))
    return hist

def compute_log_bin(dist_i,dist_bins):
    x_i=(np.log(dist_i)+2.0)/3.0
    return np.floor(dist_bins*x_i)

def compute_tan_bin(dist_i,dist_bins):
    x_i=np.log(dist_i)
    x_i=np.tan(dist_i)/(np.pi/2.0)
    x_i=(x_i+1.0)/2.0
    return np.floor(dist_bins*x_i)

def get_vectors(key_point,points):
    return [ point_i-key_point
             for point_i in points]	

def get_distances(vectors):
    return [ np.linalg.norm(vector_i)
             for vector_i in vectors]	

def make_point(x,y):
    return np.array([x,y],dtype=float) 

def compare(img1,img2):
    hist1=get_shape_context(img1)
    hist2=get_shape_context(img2)
    print(np.linalg.norm(hist1-hist2))

if __name__ == "__main__":
    #get_shape_context("img2.jpg")
    compare("img1.jpg","img2.jpg")