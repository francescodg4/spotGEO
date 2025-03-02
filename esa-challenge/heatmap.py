import numpy as np
from skimage import filters


class Heatmap(object):
    def __init__(self, sigma):
        self.sigma = sigma

        
    def from_coords(self, coords):
        '''Returns heatmap from list of coordinates'''
        Idelta = np.zeros((480, 640))    

        for r, c in coords:
            Idelta[r, c] = 1
            
        return self.from_delta(Idelta)

    
    def from_delta(self, Idelta):
        '''Returns normed mixture of gaussians centered at delta locations'''
        A = (2*np.pi*self.sigma*self.sigma)
        G = filters.gaussian(Idelta, self.sigma)
        return np.clip(A*G, 0, 1)
