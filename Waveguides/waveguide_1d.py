import sys, os
path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(path)))

from units import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from scipy import sparse 
from scipy.sparse import linalg

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
        
        res = root_scalar(waveguide_fun,bracket=(alpha_low*0.999+alpha_high*0.001,alpha_high*0.999+alpha_low*0.001),method = "brenth")
        
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
        ax.axvline(x=0, color='green', linestyle='--', linewidth=1.5,label = f"center")
        ax.set_xlabel("x/um")
        
        handles, labels = ax.get_legend_handles_labels()
        handles1, labels1 = ax1.get_legend_handles_labels()
        ax.legend(handles=handles + handles1, labels=labels + labels1, loc='upper right')
        
        ax.set_title(f"{self.mode_num}-th mode, with Neff = {Neff:.6f}")
        plt.show()

class waveguide_1d(waveguide_1d_plot):
    def __init__(self,lbd0,n_fun,x_bound,args = (),N_mesh = 640):
        # self.hx = (x_bound[1] - x_bound[0])/N_mesh
        # xlist_r = np.arange(self.hx,x_bound[1],self.hx)
        # xlist_l = np.arange(0,-x_bound[0]+self.hx,self.hx)
        # self.xlist = np.concatenate((-xlist_l[::-1],xlist_r),axis = 0) 
        
        self.xlist = np.linspace(x_bound[0],x_bound[1],N_mesh,endpoint = True,dtype = np.longdouble)
        self.hx = self.xlist[1] - self.xlist[0]
        
        self.n_x = n_fun(self.xlist,*args)
        self.k0 = 2*pi/lbd0
        
        self.mat_d = (-2*np.eye(N_mesh) + np.eye(N_mesh,k=1) + np.eye(N_mesh,k=-1))/(self.hx**2)
        mat_c = np.zeros_like(self.mat_d)
        mat_c[np.arange(N_mesh),np.arange(N_mesh)] = (self.n_x*self.k0)**2
    
        F = self.mat_d + mat_c
        self.F = sparse.dia_matrix(F)
    
    def waveguide(self,mode_num = 0):
        eig_val, eig_vec = linalg.eigsh(self.F,k = mode_num+1,which = "LA")
        self.mode_num = mode_num
        
        Neff = np.sqrt(eig_val[-(1+mode_num)])/self.k0
        field = eig_vec[:,-(1+mode_num)]
        
        self._visualization(self.n_x,field,Neff)

class nonlinear_waveguide_1d(waveguide_1d):
    def __init__(self,lbd0,n_fun,x_bound,args = (),N_mesh = 480):
        waveguide_1d.__init__(self,lbd0,n_fun,x_bound,args,N_mesh)
        self.L = x_bound[1] - x_bound[0]
        self.N = N_mesh
        
    def waveguide_n(self,mode_num = 0,delta_n_fun = lambda x: 0,E2_norm = 1e2,iter = 10):
        self.mode_num = mode_num
        self.delta_n_fun = delta_n_fun
        self.E2_norm = E2_norm
        _, self.eig_vec = linalg.eigsh(self.F,k = mode_num+1,which = "LA")
        
        n1 = iter//3 + 1
        i_amp = np.linspace(0.2,1,n1)
        for i1 in range(n1):
            self._iteration(i_amp[i1])
        
        n2 = 2*iter//3 + 1
        for _ in range(n2):
            self._iteration(1)
        
        eig_val, eig_vec = linalg.eigsh(self.F,k = mode_num+1,which = "LA")
        
        Neff_n = np.sqrt(eig_val[-(1+mode_num)])/self.k0
        field_n = eig_vec[:,-(1+mode_num)]
        
        E_inte = np.sum(np.abs(field_n)**2,axis = None)*self.hx/self.L
        amp = np.sqrt(self.E2_norm/E_inte)
        
        self._visualization(self.n_x_n,field_n*amp,Neff_n)
        
    def _iteration(self,i_amp = 1):
        vec = self.eig_vec[:,-(1+self.mode_num)]
        E2_ave = np.sum(np.abs(vec)**2,axis = None)*self.hx/self.L
        
        amp = np.sqrt(self.E2_norm/E2_ave)*i_amp

        delta_n = self.delta_n_fun(self.xlist,vec*amp)
        self.n_x_n = delta_n + self.n_x
        mat_c_n = np.zeros_like(self.mat_d)
        mat_c_n[np.arange(self.N),np.arange(self.N)] = self.n_x_n
        
        F = self.mat_d + (mat_c_n*self.k0)**2
        self.F = sparse.dia_matrix(F)
        
        _, self.eig_vec = linalg.eigsh(self.F,k = 1+self.mode_num,which = "LA")

def main():
    n1 = 1.5
    n2 = 1.2
    dis = 1.6*um
    a = 1*um
    lbd = 780*nm
    x_bound = (-4.5*um,4.5*um)
    n_fun = lambda x: n2 + (n1-n2)*(np.abs(x - dis/2)<=a/2) + (n1-n2)*(np.abs(x + dis/2)<=a/2)
    
    wvg = waveguide_1d(lbd,n_fun,x_bound,N_mesh = 1280)
    wvg.waveguide(mode_num = 2)
    
    # wvg = waveguide_1d_plot(lbd,n1,n2,a)
    # wvg.calculate(mode_num=0)

def main_n():
    n1 = 1.5
    n2 = 1.3
    a = 2.*um
    lbd = 780*nm
    x_bound = (-4*um,4*um)
    n_fun = lambda x: n2 + (n1-n2)*(np.abs(x)<=a/2)
    
    d = -1e-7
    delta_n_fun = lambda x,E: d * np.abs(E)**2*(np.abs(x)<=a/2)
    E2_norm = 6e4
    
    wvg = nonlinear_waveguide_1d(lbd,n_fun,x_bound)
    wvg.waveguide_n(mode_num = 0,delta_n_fun = delta_n_fun, E2_norm = E2_norm,iter = 70)

if __name__ == "__main__":
    main()
    # main_n()
        