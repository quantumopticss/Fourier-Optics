import sys, os
path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(path)))

from units import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from Waveguides.waveguide_1d import waveguide_1d
from scipy import sparse
from scipy.sparse import linalg
    
class soliton_1d(waveguide_1d):
    def __init__(self,lbd0,n_fun_0,x_bound,args = (),N_mesh = 640):
        waveguide_1d.__init__(self,lbd0,n_fun_0,x_bound,args,N_mesh)
        _, self.eig_vec = eigh(self.F)
        self.L = x_bound[1] - x_bound[0]
        self.N = N_mesh
        
    def soliton(self,n_fun_xe:lambda x,e: 0,mode_num = 0,E2_norm = 1e4,iter = 10):
        self.mode_num = mode_num
        self.n_fun_xe = n_fun_xe
        self.E2_norm = E2_norm
        self.field = self.eig_vec[:,-(1+mode_num)]
        
        n1 = iter//2 + 1
        f_amp = np.linspace(0.5,1.005,n1)
        n_amp = np.linspace(0.2,1.005,n1)
        for i in range(n1):
            self._iteration(f_amp[i],n_amp[i])
        
        n2 = iter//2 + 1
        for _ in range(n2):
            self._iteration(1.,1.)

        eig_val, eig_vec = eigh(self.F)
        beta = np.sqrt(eig_val[-(1+mode_num)])
        field = eig_vec[:,-(1+mode_num)]
        
        E2_ave = np.sum(np.abs(field)**2,axis = None)*self.hx/self.L
        amp = np.sqrt(self.E2_norm/E2_ave)
        Neff = beta/self.k0
        
        nlist = n_fun_xe(self.xlist,field*amp)
        self._visualization(nlist,field*amp,Neff)
        
    def _iteration(self,f_amp = 1,n_amp = 1):
        E2_ave = np.sum(np.abs(self.field)**2,axis = None)*self.hx/self.L
        amp = f_amp*np.sqrt(self.E2_norm/E2_ave)
        
        delta_n = self.n_fun_xe(self.xlist,self.field*amp)
        n_iter = (delta_n - self.n_x)*n_amp + self.n_x
        
        self.F = np.copy(self.mat_d)
        self.F[np.arange(self.N),np.arange(self.N)] = self.F[np.arange(self.N),np.arange(self.N)] + (n_iter*self.k0)**2
        
        _, eig_val = eigh(self.F)
        self.field = eig_val[:,-(1+self.mode_num)]

class soliton_1d_plot:
    """
    n = n0 + n2*I, I = 0.5*np.abs(E)**2/eta
    d = 0.5*n2/eta
    """
    def __init__(self,lbd,n,d,A0,x_bound = (-10*um,10*um)):
        self.xlist = np.linspace(x_bound[0],x_bound[1],200,endpoint = False)
        k0 = 2*pi/lbd
        W0 = np.sqrt(n/(d*k0**2*A0**2))
        
        self.A_x = A0/np.cosh(self.xlist/W0)
        self.n_x = n + d*np.abs(self.A_x)**2
        z0 = pi*W0**2/lbd
        self.Neff = n + 1/(4*z0*k0)
        
    def soliton_plot(self):
        fig, ax = plt.subplots()
        ax.plot(self.xlist*1e6,self.A_x,label = "field distribution - soliton",color = "r")
        ax1 = ax.twinx()
        ax1.plot(self.xlist*1e6,self.n_x,label = "refractive index",color = "b")
        
        handles, labels = ax.get_legend_handles_labels()
        handles1, labels1 = ax1.get_legend_handles_labels()
        ax.legend(handles=handles + handles1, labels=labels + labels1, loc='upper right')
        
        ax.set_xlabel("x/um")
        ax.set_title(f"plot soliton 0-th mode, Neff = {self.Neff:.6f}")
        
        plt.show()

def main_soliton():
    n1 = 1.5
    n2 = 1.3
    a = 2.*um
    lbd = 780*nm
    x_bound = (-20*um,20*um)
    n_fun = lambda x: n2 + (n1-n2)*(np.abs(x)<=a/2)
    
    d = 3*1e-8
    n_fun_xe = lambda x,E: n1 + d*np.abs(E)**2
    E2_norm = 0.51*1e4
    
    wvg = soliton_1d(lbd,n_fun,x_bound,N_mesh = 600)
    wvg.soliton(mode_num = 0,n_fun_xe = n_fun_xe, E2_norm = E2_norm,iter = 32)

def plot_soliton():
    n1 = 1.5
    lbd = 780*nm
    x_bound = (-20*um,20*um)
    
    d = 3*1e-8
    E2_norm = 174.4**2
    
    sl = soliton_1d_plot(lbd,n1,d,np.sqrt(E2_norm),x_bound)
    sl.soliton_plot()

if __name__ == "__main__":
    # main_soliton()
    plot_soliton()