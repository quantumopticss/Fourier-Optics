import sys,os
path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(path)))

import numpy as np
from units import *
import matplotlib.pyplot as plt

class gaussian_beam:
    ## 3D gaussian beam
    def __init__(self,lbd = 780*nm,z_bound = (-10*um,10*um),rho_bound = (-10*um,10*um), dimension = 3):
        self.lbd = lbd
        self.nu = c0_const/lbd
        self.k0 = 2*pi/lbd
        self.dimension = dimension
        
        zlist = np.linspace(z_bound[0],z_bound[1],500)
        rho_list = np.linspace(rho_bound[0],rho_bound[1],500)
        
        self.dz = zlist[1] - zlist[0]
        self.dr = rho_list[1] - rho_list[0]
        
        self.zz, self.rr = np.meshgrid(zlist,rho_list)
        
    def gaussian_beam(self,w0 = 3*um,z = 1*um, E0 = 100*V/m):
        self.w0 = w0
        self.z0 = pi*w0**2/self.lbd
        plot_zz = self.zz - z
        
        self.R = plot_zz + self.z0**2/plot_zz
        self.W = w0*np.sqrt(1 + (plot_zz/self.z0)**2)
        
        if self.dimension == 3:
            self.E = E0*w0/self.W * np.exp(-(self.rr/self.W)**2) * np.exp(-1j*self.k0*(self.zz + self.rr**2/(2*self.R)) + 1j*np.arctan(self.zz/self.z0))
            
        elif self.dimension == 2:
            self.E = E0*np.sqrt(w0/self.W) * np.exp(-(self.rr/self.W)**2) * np.exp(-1j*self.k0*(self.zz + self.rr**2/(2*self.R)) + 0.5*1j*np.arctan(self.zz/self.z0))
        
        return self
        
    def filtering(self,filter = lambda nu: 1.):
        self.dz
        
        
        
        
    def _visualization(self):
        plt.figure(1)
        plt.imshow(np.abs(self.E))
        plt.colorbar()
        plt.title("Enorm @ gaussian beam")
        
        plt.show()
        
def main():
    gb = gaussian_beam(dimension = 3)
    gb.gaussian_beam()
    
if __name__ == "__main__":
    main()