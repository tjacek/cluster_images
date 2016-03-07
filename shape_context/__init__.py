import cv2
import numpy as np

def get_shape_context(in_path):
    points=read_points(in_path)
    print(len(points))

def read_points(in_path):
    img=cv2.imread(in_path,cv2.IMREAD_GRAYSCALE)
    points=[]
    for (x,y), value in np.ndenumerate(img):
      if(value!=0):
          points.append(make_point(x,y))
    return points

def make_point(x,y):
    return np.array([x,y]) 

if __name__ == "__main__":
    get_shape_context("in.jpg")