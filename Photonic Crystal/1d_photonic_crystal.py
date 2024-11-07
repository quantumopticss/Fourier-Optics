"""
1D photonic crystal
"""

import sys, os
path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(path)))

from units import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh, eig

class photonic_crystal_1d:
    def __init__(self,eta_fun,gamma,N = 512,mode = "TM"):
        match N%4:
            case 3:
                N += 1
            case 2:
                N += 2
            case 1:
                N += 3
                
        self.N = N
        self.x_list = np.linspace(0.,gamma,N,endpoint = False)
        self.eta_list = eta_fun(self.x_list)
        self.g = 2*pi/gamma
        self.f_eta = np.fft.fftshift(np.fft.fft(self.eta_list))/N
        
        eta_mat = np.empty([N//2,N//2],dtype = np.complex128)
        L = np.arange(-N//4,N//4)
        for i in range(N//2):
            eta_mat[i,:] = self.f_eta[L[i]-L+N//2]
            
        self.L_g = L*self.g
        self.eta_mat = eta_mat

    def _calculate(self,k_bloch,kx):
        
        L_g = self.L_g + k_bloch 
        mat_c = np.tensordot(L_g,L_g,axes = 0)
        
        F = self.eta_mat*( kx**2 + mat_c)
        eig_val, eig_vec = eigh(F)
        return eig_val, eig_vec
    
    def _general_k_series(self,x):
        kb = x*(x<1.)*(x>=0.) + 1.*(x>=1.)*(x<2.) + (3.-x)*(x>=2.)*(x<=3.)
        kx = 0.*(x<1.)*(x>=0.) + (x-1.)*(x>=1.)*(x<2.) + (3.-x)*(x>=2.)*(x<=3.)

        return kb*self.g, kx*self.g ## may be more about kx

    def photonic_bands(self,band_num = 3):
        genlist = np.linspace(0.001,0.999,80)
        kb_list, kx_list = self._general_k_series(genlist)
        
        eig_val, _ = self._calculate(kb_list[0],kx_list[0])
        
        self.visual = np.array([eig_val[0:band_num]])
        for i in range(1,len(genlist)):
            eig_val, _ = self._calculate(kb_list[i],kx_list[i])
            omega = eig_val[0:band_num]
            self.visual = np.concatenate((self.visual,[omega]),axis = 0)
            
        self.visual = c0_const*np.sqrt(self.visual) # omega
        
        plt.figure(1)
        for i in range(band_num):
            plt.plot(genlist,self.visual[:,i],label = f"w-k relation - {i}")
        plt.ylabel("omega")
        plt.xlabel("photonic k space")
        plt.title('photonic dispersion relation')
        plt.legend()
        
        plt.show()
    
    def fields(self,p_kb,p_k0,mode_num = 0):
        eig_val, eig_vec = self._calculate(p_kb*self.g,p_k0*self.g)
        vec = eig_vec[:,mode_num]
        
        freq = c0_const*np.sqrt(eig_val[mode_num])/(1e12*2*pi)
        H_field = np.fft.ifftshift(np.fft.ifft(vec))*self.N*np.exp(-1j*p_kb*self.g*self.x_list[0:self.N//2])
        
        plt.figure(2)
        plt.plot(self.x_list[self.N//4:3*self.N//4]*1e6,np.abs(H_field),label = f"{mode_num}th - field @ normH")
        plt.plot(self.x_list[self.N//4:3*self.N//4]*1e6,self.eta_list[self.N//4:3*self.N//4],label = "eta - x")
        plt.xlabel("x/um")
        plt.ylabel("normH")
        plt.title(f"H_field, freq = {freq:.3f} THz")
        plt.legend()
        
        plt.show()
        
def main():
    
    n1 = 1.5
    n2 = 3.2
    d1 = 1*um
    d2 = 1*um
    
    n_fun = lambda x: n1 + (n2-n1)*(x>d1)*(x<=(d2+d1))
    
    eta_fun = lambda x: 1/(n_fun(x)**2)
    gamma = d1 + d2
    
    pc = photonic_crystal_1d(eta_fun,gamma)
    pc.photonic_bands(band_num = 4)
    # pc.fields(0.55,0.,mode_num = 1)
    
if __name__ == "__main__":
    main()
