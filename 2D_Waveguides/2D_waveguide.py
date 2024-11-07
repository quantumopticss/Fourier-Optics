import sys, os
path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(path)))

from units import *
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import linalg

class waveguide_2d:
    """we use eigvalue method to solve 2D waveguides which are difficult to analytically solve   
    but there is a problem that if we user a more finess mesh, the solver will return a wrong solution, and 
    the returned N_eff is always hihgher than correct one 
    """
    def __init__(self,lbd,n_fun_xy,args = (),x_bound = (-5*um,5*um),y_bound = (-5*um,5*um),N_mesh = 128):
        self.N_mesh = N_mesh
        
        xlist = np.linspace(x_bound[0],x_bound[1],N_mesh,endpoint = False)
        ylist = np.linspace(y_bound[0],y_bound[1],N_mesh,endpoint = False)
        
        self.xx, self.yy = np.meshgrid(xlist,ylist)
        
        self.hx = xlist[1] - xlist[0]
        self.hy = ylist[1] - ylist[0]
        
        mat_2d = ( np.eye(N_mesh**2,k = N_mesh) + np.eye(N_mesh**2,k = - N_mesh) )/self.hx**2
        mat_2d_make = -2*np.eye(N_mesh)*(1/self.hx**2 + 1/self.hy**2) + ( np.eye(N_mesh,k=1) + np.eye(N_mesh,k=-1) )/self.hy**2
        
        for i in range(N_mesh):
            mat_2d[i*N_mesh:(i+1)*N_mesh,i*N_mesh:(i+1)*N_mesh] = mat_2d_make
        
        
        # mat_2d = ( 16.*np.eye(N_mesh**2,k = N_mesh) + 16.*np.eye(N_mesh**2,k = - N_mesh) - 
        #           np.eye(N_mesh**2,k = 2*N_mesh) - np.eye(N_mesh**2,k = - 2*N_mesh))/(12*self.hx**2)
        
        # mat_2d_make = ( -15.*np.eye(N_mesh)*(1/self.hx**2 + 1/self.hy**2))  + (
        #                16.*( np.eye(N_mesh,k=1) + np.eye(N_mesh,k=-1) ) - ( np.eye(N_mesh,k=2) + np.eye(N_mesh,k=-2) )
        #                )/(12*self.hy**2)
        
        for i in range(N_mesh):
            mat_2d[i*N_mesh:(i+1)*N_mesh,i*N_mesh:(i+1)*N_mesh] = mat_2d_make
        
        self.n_xy = n_fun_xy(self.xx,self.yy,*args)
        n_xy_reshape = np.reshape(self.n_xy,-1,order = "F")
        
        self.k0 = 2*pi/lbd
        
        mat_k = np.zeros_like(mat_2d)
        mat_k[np.arange(N_mesh**2),np.arange(N_mesh**2)] = n_xy_reshape
        
        self.Mat = mat_2d + (mat_k*self.k0)**2
        self.Mat_s = sparse.csr_matrix(self.Mat)
    
    def waveguide(self,mode_num:list = [0,1,2]):
        N = np.max(np.array(mode_num))
        num = len(mode_num)
        # eig_val, eig_vec = eigh(self.Mat)
        eig_val, eig_vec = linalg.eigsh(self.Mat_s,k = N + 1,which = "LM")
        
        for i in np.arange(num):
            beta = np.sqrt(eig_val[mode_num[i]])
            N_eff = beta/self.k0
            field = eig_vec[:,mode_num[i]]
            field = np.reshape(field,[self.N_mesh,self.N_mesh],order = "F")
            plt.figure(i)
            plt.imshow(np.abs(np.abs(field)),cmap = "rainbow")
            plt.colorbar()
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title(f"effective mode index = {N_eff:.4f}")
        
        plt.figure(num)
        plt.imshow(self.n_xy)
        plt.title("refractive index")
        plt.colorbar()
        
        plt.show()

def main():
    x_bound = (-3*um,3*um)
    y_bound = (-4*um,4*um)
    
    a = 2*um
    b = 3*um
    c = 1.2*um
    d = 1*um
    
    n1 = 1.5
    n2 = 1.2
    
    n_fun =  lambda x,y : n2 + (n1 - n2)*(np.abs(x) <= b/2)*(np.abs(y) <= a/2) + (n1-n2)*(np.abs(y-a/2-d/2)<=d/2)*(np.abs(x) <= c/2) # + (n1-n2)*(np.abs(y+a/2+d/2)<=d/2)*(np.abs(x) <= c/2)
    # n_fun =  lambda x,y : n2 + (n1 - n2)*(np.abs(x) <= b/2)*(np.abs(y) <= a/2)
    # n_fun = lambda x,y : n2 + (n1-n2)*(np.sqrt(x**2 + y**2) <= a )
    
    lbd = 780*nm
    wvg = waveguide_2d(lbd,n_fun,x_bound,y_bound,N_mesh = 32)
    wvg.waveguide(mode_num = [0,1,2,3])
    
if __name__ == "__main__":
    main()         

        
        