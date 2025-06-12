import sys, os
path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(path))))

import numpy as np
from numpy import fft
from units import *
import matplotlib.pyplot as plt
from propagation_fresnel import propagate

class UL_shift:
    def __init__(self,xtfunc,nu_shift = 2*pi*c0_const/(13.5*nm),x_tuple = (-20*um,20*um,300),t_tuple = (0,6*fs,120),c = c0_const,args = ()):
        
        ## spacial domain
        x0, xe, Nx = x_tuple
        self.xlist, self.dx = np.linspace(x0,xe,Nx,endpoint=False,retstep=True)
        self.f_spacial = np.arange(-Nx//2,Nx//2)/(xe-x0)
        
        ## time domain
        t0, te, Nt = t_tuple
        self.tlist, self.dt = np.linspace(t0,te,Nt,endpoint=False,retstep=True)
        self.f_time = np.arange(-Nt//2,Nt//2)/(te-t0) + nu_shift
        self.lbd_list = c/self.f_time
        
        ## couple time and space
        self.xx, self.tt = np.meshgrid(self.xlist,self.tlist,indexing="ij")
        self.fxx, self.ftt = np.meshgrid(self.f_spacial,self.f_time,indexing = "ij")
        
        self.xtfun = xtfunc(self.xx, self.tt, *args)
        self.ft_xtfun = fft.fftshift(fft.fft(self.xtfun,axis = 1),axes = 1) / Nt
        
    def UL_shift(self,d = 3*m,c = c0_const):
        self.N, self.M = self.ft_xtfun.shape # N - x, M - time
        self.ft_propagate = np.empty_like(self.ft_xtfun)
        for i in range(self.M):
            print(i)
            lbd_i = c/self.f_time[i]
            self.ft_propagate[:,i] = propagate(self.ft_xtfun[:,i],lbd = lbd_i,d = d,dx = self.dx)
            
        self.UL_xtfun = fft.ifft(self.ft_propagate,axis = 1) / self.dt
        
        return self
        
    def visualization(self,amp = 0.4):
        fig, ax = plt.subplots()
        P1 = np.abs(self.ft_propagate[self.N//2,:])**2
        P2 = np.abs(self.ft_propagate[self.N//2 + int(self.N*amp),:])**2
        P3 = np.abs(self.ft_propagate[self.N//2 + int(self.N*amp/2),:])**2
        lbd_list = c0_const/self.f_time * 1e9
        ax.plot(lbd_list,P1/np.max(P1),label = "on axis point")
        ax.plot(lbd_list,P2/np.max(P2),label = f"off axis point, with dis = {amp:.2f}")
        ax.plot(lbd_list,P3/np.max(P3),label = f"off axis point, with dis = {amp/2:.2f}")
        ax.set_xlabel("wavelength-nm")
        ax.set_ylabel("normalized power spectrum")
        ax.set_title("UL off axis red shift")
        ax.legend()
        
        plt.show()
        
x0 = 4*um
tau = 0.5*fs; t0 = 3*fs

t1 = 0.05*fs
nu0 = 1/t1

def xt_func(x,t):
    mat = np.exp(-(x/x0)**2) * np.exp(-((t - t0)/tau)**2)
        
    return mat

test = UL_shift(xt_func,nu0)
test.UL_shift(d = 1*m).visualization()
