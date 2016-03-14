import numpy as np

class Pcloud(object):
    def __init__(self,points):
        self.points=points
        self.dims=points[0].shape[0]
       
    def __getitem__(self,index):
        return self.seq[index]

    def __str__(self): 
        return str(len(self.seq))+" "+str(self.cat)

    def __len__(self):
        return len(self.points) 

    def get_numpy(self):
        return np.array(self.points)


def make_point_cloud(img,dim3D=True):
    points=[]
    if(dim3D):
        for (x, y), element in np.ndenumerate(img):
            z=img[x][y]
            if(z!=0):
                points.append(make_point3D(x,y,z))
    else:
        for (x, y), element in np.ndenumerate(img):
            points.append(make_point2D(x,y))
    if(len(points)==0):
        return None
    return Pcloud(points)

def make_point3D(x,y,z):
    return np.array([x,y,z],dtype=float)

def make_point2D(x,y):
    return np.array([x,y],dtype=float)	