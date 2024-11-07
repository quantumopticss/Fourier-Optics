import sys, os
path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(path)))

from units import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from scipy.linalg import eigh

"""
helmholtz equation (\nabla^2 + n^2*k0^2)U = 0
for waveguide mode, U = f(x)*np.exp(-1j*beta*z)
"""

class waveguide_1d_plot:
    def __init__(self,lbd,n1,n2,a):
        self.n1 = n1
        self.NA = np.sqrt(n1**2 - n2**2)
        self.a = a
        self.lbd = lbd
        self.k0 = 2*pi/lbd
        self.xlist = np.linspace(-2*a,2*a,400)
    
    def calculate(self,mode_num:int = 0):
        self.mode_num = mode_num
        n = mode_num//2
        if mode_num % 2 == 0:
            waveguide_fun = lambda x: x*np.tan(pi*self.a*x/self.lbd) - np.sqrt(self.NA**2 - x**2)
            alpha_low = n*self.lbd/self.a
            alpha_high = (n+1/2)*self.lbd/self.a
        else:
            waveguide_fun = lambda x: np.sqrt(self.NA**2 - x**2) + x/np.tan(pi*self.a*x/self.lbd)
            alpha_low = (n+1/2)*self.lbd/self.a
            alpha_high = (n+1)*self.lbd/self.a

        if alpha_low > self.NA:
            raise ValueError("can't guide that mode")
        
        res = root_scalar(waveguide_fun,bracket=(alpha_low*0.999+alpha_high*0.001,alpha_high*0.999+alpha_low*0.001),method = "bisect")
        
        alpha = res.root
        kt = alpha*self.k0
        gamma = np.sqrt(self.NA**2 - alpha**2)*self.k0
        
        Neff = np.sqrt(self.n1**2 - alpha**2)
        if mode_num % 2 == 0:
            field = ( np.cos(kt*self.xlist)*(np.abs(self.xlist)<=self.a/2)
                    + np.cos(kt*self.a/2)*np.exp(gamma*self.a/2)* (np.exp(-gamma*self.xlist)*(self.xlist>self.a/2) 
                                                                 + np.exp(gamma*self.xlist)*(self.xlist<-self.a/2))
                    )
        else:
            field = ( np.sin(kt*self.xlist)*(np.abs(self.xlist)<=self.a/2)
                    + np.sin(kt*self.a/2)*np.exp(gamma*self.a/2)* (np.exp(-gamma*self.xlist)*(self.xlist>self.a/2) 
                                                                 - np.exp(gamma*self.xlist)*(self.xlist<-self.a/2))
                    )
        
        n2 = np.sqrt(self.n1**2 - self.NA**2)
        nlist = n2 + (self.n1 - n2)*(np.abs(self.xlist)<=self.a/2)
        self._visualization(nlist,field,Neff)
        
    def _visualization(self,nlist,field,Neff):
        fig, ax = plt.subplots()
        ax.plot(self.xlist*1e6,np.abs(field),label = "field distribution",c = "r")
        ax1 = ax.twinx()
        ax1.plot(self.xlist*1e6,nlist,label = "refractive index",c = "b")
        ax.legend()
        ax.set_xlabel("x/um")
        ax.set_title(f"{self.mode_num}-th mode, with Neff = {Neff:.4f}")
        plt.show()

class waveguide_1d_calculate(waveguide_1d_plot):
    def __init__(self,lbd0,n_fun,x_bound,args = (),N_mesh = 400):
        self.xlist = np.linspace(x_bound[0],x_bound[1],N_mesh,endpoint = False)
        self.hx = (x_bound[1] - x_bound[0])/N_mesh
        
        self.n_x = n_fun(self.xlist,*args)
        self.k0 = 2*pi/lbd0
        
        mat_d = (-2*np.eye(N_mesh) + np.eye(N_mesh,k=1) + np.eye(N_mesh,k=-1))/(self.hx**2)
        mat_c = np.zeros_like(mat_d)
        mat_c[np.arange(N_mesh),np.arange(N_mesh)] = (self.n_x*self.k0)**2
    
        self.F = mat_d + mat_c
    
    def waveguide(self,mode_num = 0):
        eig_val, eig_vec = eigh(self.F)
        self.mode_num = mode_num
        
        Neff = np.sqrt(eig_val[-(1+mode_num)])/self.k0
        field = eig_vec[:,-(1+mode_num)]
        
        self._visualization(self.n_x,field,Neff)
        
def main():
    n1 = 1.5
    n2 = 1.3
    a = 2.*um
    lbd = 780*nm
    x_bound = (-4*um,4*um)
    n_fun = lambda x: n2 + (n1-n2)*(np.abs(x)<=a/2)
    
    wvg = waveguide_1d_calculate(lbd,n_fun,x_bound)
    wvg.waveguide(mode_num = 0)
    
    # wvg = waveguide_1d_plot(lbd,n1,n2,a)
    # wvg.calculate(mode_num=0)

if __name__ == "__main__":
    main()
        