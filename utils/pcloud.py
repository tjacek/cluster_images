import numpy as np

class Pcloud(object):
    def __init__(self,points):
        self.points=np.array(points)
        self.len=len(points)
        self.dims=self.points.shape[1]
        #self.points=points
        #self.dims=points[0].shape[0]
       
    def __getitem__(self,index):
        return self.points[index]

    def __str__(self): 
        return str(len(self.seq))+" "+str(self.cat)

    def __len__(self):
        return self.len 

    def get_numpy(self):
        return self.points#np.array(self.points)

    def center_of_mass(self):
        return np.mean(self.points,axis=0)

    def max(self,dim):
        index=np.argmax(self.points[:,dim],axis=0)
        print(index)
        return self.points[index]
   
    def min(self,dim):
        index=np.argmin(self.points[:,dim],axis=0)
        return self.points[index]

def make_point_cloud(img,dim3D=True):
    img=img.get_orginal()
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

def normalized_cloud(pcloud):
    dim,min_dim=get_dim(pcloud)
    norm=np.linalg.norm(dim)
    new_points=[ (point_i-min_dim)/norm
                 for point_i in pcloud.points]
    return Pcloud(new_points)

def unit_normalized(pcloud):
    dim,min_dim=get_dim(pcloud)
    #norm=np.linalg.norm(dim)
    new_points=[ (point_i-min_dim)/dim
                 for point_i in pcloud.points]
    return Pcloud(new_points)

def make_point3D(x,y,z):
    return np.array([x,y,z],dtype=float)

def make_point2D(x,y):
    return np.array([x,y],dtype=float)

def get_dim(pcloud):
    arr=pcloud.get_numpy()
    max_dim=np.max(arr,axis=0)
    min_dim=np.min(arr,axis=0)
    dim=max_dim-min_dim
    return dim,min_dim