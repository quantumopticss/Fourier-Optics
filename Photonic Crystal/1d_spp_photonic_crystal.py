"""
1D photonic crystal
"""

import sys, os
path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(path)))

from units import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh, eig, ishermitian

class photonic_crystal_1d:
    def __init__(self,eta_fun,gamma,N = 128,mode = "TM",**kwargs):
        self.mode = mode
        match N%4:
            case 3:
                N += 1
            case 2:
                N += 2
            case 1:
                N += 3
                
        self.N = N
        self.x_list = np.linspace(0.,gamma,N,endpoint = False,dtype = np.longdouble)
        self.eta_list = eta_fun(self.x_list)
        self.g = 2*pi/gamma
        self.f_eta = np.fft.fftshift(np.fft.fft(self.eta_list))/N
        self.kx_max = kwargs.get("kx_max",self.g)

        eta_mat = np.empty([N//2,N//2],dtype = np.longcomplex)
        L = np.arange(-N//4,N//4)
        for i in range(N//2):
            eta_mat[i,:] = self.f_eta[L[i]-L+N//2]
        
        self.L_g = L*self.g
        self.eta_mat = eta_mat
    
    def photonic_crystal(self,omega_num = 3,mode_num = 0):
        sweep_list = np.linspace(1e-3,0.99,49,endpoint = False)
        
        kb_list = 0
        kx_list = sweep_list*
        
        omega_list = np.empty([len(sweep_list),omega_num])
        
        for i in range(len(sweep_list)):
            omega_cal, _ = self._photonic_crystal_cal(kb_list[i],kx_list[i],omega_num,mode_num)
            omega_list[i,:] = np.real(omega_cal)
     
        self.visualization(omega_list,sweep_list)
    
    def _photonic_crystal_cal(self,kb,kx,omega_num,mode_num):
        if self.mode == "TM":
            self._photonic_crystal_TM(kb,kx)
            vec = self.eig_vec[:,mode_num]
            field_vec = np.fft.ifftshift(np.fft.ifft(vec))*(self.N//2)
            field = [field_vec]
        elif self.mode == "TE":
            self._photonic_crystal_TE(kb,kx)
            vec = self.eig_vec[:,mode_num]
            field_vec_hx = np.fft.ifftshift(np.fft.ifft(vec[0:self.N//2]))*(self.N//2)
            field_vec_hz = np.fft.ifftshift(np.fft.ifft(vec[self.N//2::]))*(self.N//2)
            
            field = [field_vec_hx,field_vec_hz]
        
        omega_list = c0_const*np.sqrt(self.eig_val[0:omega_num])
        return omega_list, field
    
    def _photonic_crystal_TM(self,kb,kx):
        M_pc = np.empty([self.N//2,self.N//2],dtype = np.longcomplex)
        M_pc = self.eta_mat * ( kx**2 +  np.tensordot((self.L_g + kb),(self.L_g + kb),axes = 0) )
        self.eig_val, self.eig_vec = eigh(M_pc)
    
    def _photonic_crystal_TE(self,kb,kx):
        M_pc = np.empty([self.N,self.N],dtype = np.longcomplex)
        M_pc[0:self.N//2,0:self.N//2] = self.eta_mat * np.tensordot((self.L_g+kb),(self.L_g+kb),axes = 0)
        M_pc[0:self.N//2,self.N//2::] = -kx*self.eta_mat * (self.L_g[np.newaxis,:])
        M_pc[self.N//2::,0:self.N//2] = -kx*self.eta_mat * (self.L_g[np.newaxis,:])
        M_pc[self.N//2::,self.N//2::] = kx**2*self.eta_mat
        self.eig_val, self.eig_vec = eig(M_pc)

    def visualization(self,omega_list,sweep_list):
        _, n = omega_list.shape
        
        fig, ax = plt.subplots()
        for i in range(n):
            ax.plot(sweep_list,omega_list[:,i],label = "{i} - th, photonic band")
            
        ax.set_xlabel("photonic sweep")
        ax.set_ylabel("omega")
        ax.set_title("photonic band - " + self.mode)
        
        plt.show()
        
def main():
    n1 = 1.5
    n2 = 1.2
    d1 = 1*um
    d2 = 1.2*um
    
    n_fun = lambda x: n1 + (n2-n1)*(x>d1)*(x<=(d2+d1))
    
    eta_fun = lambda x: 1/(n_fun(x)**2)
    gamma = d1 + d2
    
    pc = photonic_crystal_1d(eta_fun,gamma,mode = "TM",N = 512)
    pc.photonic_crystal(omega_num = 3)
    
if __name__ == "__main__":
    main()
