import sys, os
path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(path)))

import numpy as np
from scipy.sparse import linalg, dia_matrix
from units import *
import matplotlib.pyplot as plt

# u'' + (k**2 - (L**2 - 0.25)/rho**2 )*u = beta**2*u # suitable for L != 0 condition

class fiber:
    def __init__(self,nfun,args:tuple = (),lbd:float = 780*nm,r_max:float = 1*um,N_mesh:int = 1024):
        self.rho_list,self.h = np.linspace(0.1*nm,r_max,N_mesh,retstep=True)
        self.k0 = 2*pi/lbd
        self.lbd = lbd
        self.n_list = nfun(self.rho_list,*args)   
        self.N_mesh = N_mesh
        
    def mode_cal(self,L:int = 1,mode_num:int = 0):
        ## (1, -2, 1)/(h**2)
        self.mode_num = mode_num
        self.L = L
        data = np.ones([3,self.N_mesh])/self.h**2 
        data[1,:] = data[1,:]*(-2.0) + (self.n_list*self.k0)**2 - (L**2 - 0.25)/self.rho_list**2
        off_set = np.array([-1,0,1])
        
        self.Mat = dia_matrix((data,off_set),shape = (self.N_mesh,self.N_mesh))
        
        self.eig_val, self.eig_vec = linalg.eigsh(self.Mat,k = (mode_num+1), which = "LA")
        
        self.n_eff = np.sqrt(self.eig_val[-(mode_num+1)])/self.k0
        self.vec = self.eig_vec[:,-(mode_num+1)]
        
    def visualization(self):
        fig, ax1 = plt.subplots()
        ax1.plot(self.rho_list*1e9,self.vec,label = f"mode_num = {self.mode_num}")
        
        ax2 = ax1.twinx()
        ax2.plot(self.rho_list*1e9,self.n_list,label = f"refractive index distribution",c = "r")
        
        ax1.set_xlabel("rho/nm")
        ax1.set_ylabel("E_field")
        ax1.set_title(f"mode_num = {self.mode_num}, L = {self.L}, Neff = {self.n_eff:.6f} @ lbd = {self.lbd*1e9:.2f}nm")  
        
        fig.legend()
        
        plt.show()

def main():
    a = 8*um # raidus of fiber core
    A = 4 # amp of solve area
    lbd = 420*nm # wavelength
    N = 1024 # mesh accuracy

    n1 = 1.5 # core
    n2 = 1.4 # cladding
    Delta = (n1**2 - n2**2)/(2*n1**2) # fiber Delta parameter
    P = 2 # GRIN fiber P index 

    L = 0 # azimuthal index
    M = 0 # fiber radical order index
    fiber_cls = "GRIN_fiber"
    ### define refractive index
    if fiber_cls == "GRIN_fiber":
        n_index = lambda rho: np.sqrt( n2**2 + (n1**2*( 1 - 2*Delta*((rho/a)**P)   ) - n2**2 )*(rho<=a) )
    elif fiber_cls == "step_index_fiber":
        n_index = lambda rho: n2 + (n1-n2)*(rho<=a)
    else:
        raise ValueError("Wrong fiber")
    
    eig = np.array([]) 
    f = fiber(n_index,lbd = lbd,r_max = a*6,N_mesh = N)
    f.mode_cal(L = L, mode_num = M)
    eig = np.hstack((eig,f.eig_val))
    
    neff = np.sqrt(eig)/f.k0
    Neff = np.sort(neff)[::-1]
    plt.plot(Neff)
    plt.xlabel("mode number")
    plt.title(fiber_cls + f" @ mode dispersion, std = {np.std(Neff)/np.mean(Neff):.4f}")
    plt.ylabel("Neff")
    plt.show()
    
    f.visualization()
    
if __name__ == "__main__":
    main()