import sys, os
path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(path))))

import numpy as np
from numpy import fft
from units import *
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

"""

cylindrical spherical lens with focus ability alone x axis
2f system, with impulse response function
h(x,x';nu) = j*nu/(c*f) * np.exp(-1j*4*pi*f*nu/c) * np.exp(1j*2*pi*nu/(c*f) * x*x' )

input U(x,t) = g(t)*p(x) , V(x,nu) = G(nu)*p(x)

-> VV(x,nu) = G(nu) * \int_{-infty}^{infty} h(x,x';nu) * p(x') dx' W
            = G(nu) * P(nu*x/(c*f)) * [ j*nu/(c*f) * np.exp(-1j*4*pi*f*nu/c) ]
            
-> UU(x,t) = FT^(-1) VV(x,nu)

"""

class ts_mapping:
    def __init__(self,g_func,t_tuple = (0,20*ps,1600),args = ()):
        
        t0, te, Nt = t_tuple
        self.tlist, self.dt = np.linspace(t0,te,Nt,endpoint = False,retstep = True)
        
        self.df_t = 1/(te-t0)
        self.f_t = np.arange(-Nt//2,Nt//2) * self.df_t # ~ 1e14
        
        self.g_time = g_func(self.tlist,*args) # temporal pulse
        self.f_g = fft.fftshift( fft.fft(self.g_time) ) * self.dt
        
    def ts_map_define(self,p_func,x_tuple = (-1500*um,1500*um,2560),args = ()):
        
        x0, xe, Nx = x_tuple
        self.xlist, self.dx = np.linspace(x0,xe,Nx,endpoint=False,retstep=True)
        
        self.df_x = 1/(xe-x0)
        self.f_x = np.arange(-Nx//2,Nx//2) * self.df_x
        
        self.p_space = p_func(self.xlist,*args)
        
        self.f_p = fft.fftshift(fft.fft(self.p_space)) * self.dx
        
        return self
        
    def ts_mapping(self,f = 100*mm, x0 = 10*mm, c = c0_const):
        ##
        # 1ps*c0_const = 0.3mm
        
        self.t_x_map = self.f_t * x0/(c*f) # 3e14 / (3*1e8 * 3)

        f_interpolate = CubicSpline(self.f_x,self.f_p) # because spacial and time fft may not match
        self.f_mapped_nu = f_interpolate(self.t_x_map) # time matched spacial freq grid
        
        print(self.f_mapped_nu)
        
        self.f_gg = self.f_g * self.f_mapped_nu * (1j*self.f_t/(c*f)) #* np.exp(-1j*4*pi*f*self.f_t/c)
        
        self.GG = fft.ifft(self.f_gg) * self.df_t * self.f_t.size
        
        return self
        
    def visual(self):
        
        fig, ax = plt.subplots()
        ax_t = ax.twinx()
        ax.plot(self.tlist*1e12,np.abs(self.GG),label = "shaped pulse",c = "r")
        ax_t.plot(self.tlist*1e12,self.g_time,label = "raw pulse")
        ax.set_ylabel("amp")
        fig.legend()
        ax.set_title("spacial-time mapping to shape a pulse")
        
        fig1, ax1 = plt.subplots()
        ax1.plot(self.xlist*1e3,self.p_space,label = "p_space")
        ax1.set_xlabel("x/mm")
        ax1.set_ylabel("amp")
        ax1.set_title("spacial map")
        ax1.legend()
        
        plt.show()
        
#%%
t0 = 10*ps    
tau = 3*ps

g_func = lambda t: np.exp(-(t-t0)**2/tau**2)
p_func = lambda x: np.exp(-(x/(100*um))**2)

ts_map = ts_mapping(g_func)
ts_map.ts_map_define(p_func)
ts_map.ts_mapping().visual()
    

        