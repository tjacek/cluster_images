import numpy as np

def high_rated_frames(imgs,criterion,threshold=0.9):
    return [img_i for img_i in imgs
                     if criterion(img_i)>threshold]	
