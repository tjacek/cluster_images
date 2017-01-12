import numpy as np

class FeatSeq(object):
    def __init__(self,conv):
        self.conv=conv

    def __call__(self,action):
        return [self.conv(img_i)  
                for img_i in action.img_seq]	
