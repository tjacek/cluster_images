import numpy as np

class Pcloud(object):
    def __init__(self,points):
        self.points=items
        self.dims=items[0].shape[0]
       
    def __getitem__(self,index):
        return self.seq[index]

    def __str__(self): 
        return str(len(self.seq))+" "+str(self.cat)

    def __len__(self):
        return len(self.seq) 

    def get_numpy(self):
        return np.array(self.points)


def make_point_cloud(img,3D=True):
	points=[]
	if(3D):
        for (x, y), element in np.ndenumerate(img):
            z=img[x][y]
            points.append(make_point3D(x,y,z))
    else:
        for (x, y), element in np.ndenumerate(img):
            points.append(make_point2D(x,y))
    return Pcloud(points)

def make_point3D(x,y,z):
    return np.array([x,y,z],dtype=float)

def make_point2D(x,y):
    return np.array([x,y],dtype=float)	