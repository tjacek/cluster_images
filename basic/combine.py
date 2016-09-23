import numpy as np 

class CombinedFeatures(object):
    def __init__(self, extractors):
        self.extractors = extractors

    def __call__(self,img_i):
        feats=[extr_i(img_i) 
    	        for extr_i in self.extractors]
    	return feats
		