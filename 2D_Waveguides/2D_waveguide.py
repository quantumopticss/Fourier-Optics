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
    """
    def __init__(self,lbd,n_fun_xy,args = (),x_bound = (-5*um,5*um),y_bound = (-5*um,5*um),N_mesh = 128):
        self.N_mesh = N_mesh
        self.args = args
        xlist = np.linspace(x_bound[0],x_bound[1],N_mesh)
        ylist = np.linspace(y_bound[0],y_bound[1],N_mesh)
        
        self.Lx = x_bound[1] - x_bound[0]
        self.Ly = y_bound[1] - y_bound[0]
        self.xx, self.yy = np.meshgrid(xlist,ylist)
        
        self.hx = xlist[1] - xlist[0]
        self.hy = ylist[1] - ylist[0]
        
        mat_2d = ( np.eye(N_mesh**2,k = N_mesh) + np.eye(N_mesh**2,k = - N_mesh) )/self.hx**2
        mat_2d_make = -2*np.eye(N_mesh)*(1/self.hx**2 + 1/self.hy**2) + ( np.eye(N_mesh,k=1) + np.eye(N_mesh,k=-1) )/self.hy**2
        
        for i in range(N_mesh):
            mat_2d[i*N_mesh:(i+1)*N_mesh,i*N_mesh:(i+1)*N_mesh] = mat_2d_make
        
        self.mat_2d = mat_2d # 
        
        # mat_2d = ( 16.*np.eye(N_mesh**2,k = N_mesh) + 16.*np.eye(N_mesh**2,k = - N_mesh) - 
        #           np.eye(N_mesh**2,k = 2*N_mesh) - np.eye(N_mesh**2,k = - 2*N_mesh))/(12*self.hx**2)
        
        # mat_2d_make = ( -15.*np.eye(N_mesh)*(1/self.hx**2 + 1/self.hy**2))  + (
        #                16.*( np.eye(N_mesh,k=1) + np.eye(N_mesh,k=-1) ) - ( np.eye(N_mesh,k=2) + np.eye(N_mesh,k=-2) )
        #                )/(12*self.hy**2)
        
        # for i in range(N_mesh):
        #     mat_2d[i*N_mesh:(i+1)*N_mesh,i*N_mesh:(i+1)*N_mesh] = mat_2d_make
        
        self.n_xy = n_fun_xy(self.xx,self.yy,*self.args)
        self.n_xy_reshape = np.reshape(self.n_xy,-1,order = "F")
        
        self.k0 = 2*pi/lbd
        
        mat_k = np.zeros_like(mat_2d)
        mat_k[np.arange(N_mesh**2),np.arange(N_mesh**2)] = self.n_xy_reshape
        
        Mat = mat_2d + (mat_k*self.k0)**2
        self.Mat_s = sparse.csr_matrix(Mat) # sparse
    
    def waveguide_2d(self,mode_num = 0):
        eig_val, eig_vec = linalg.eigsh(self.Mat_s,k = mode_num + 1,which = "LA")
        Neff = np.sqrt(eig_val[-(1+mode_num)])/self.k0
        field = eig_vec[:,-(1+mode_num)]
        self._visualization_2d(Neff,np.reshape(field,[self.N_mesh,self.N_mesh],"F"),self.n_xy)
    
    @staticmethod
    def _visualization_2d(N_eff,field,N_dis):

        plt.imshow(np.abs(field),cmap = "rainbow")
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"effective mode index = {N_eff:.5f}")
        
        plt.figure(2)
        plt.imshow(N_dis)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("refractive index")
        plt.colorbar()
        
        plt.show()

class nonlinear_waveguide_2d(waveguide_2d):
    def __init__(self,lbd,n_fun_xy,args = (),x_bound = (-5*um,5*um),y_bound = (-5*um,5*um),N_mesh = 128):
        waveguide_2d.__init__(self,lbd,n_fun_xy,args = args,x_bound = x_bound,y_bound = y_bound,N_mesh = N_mesh)
        
    def waveguide_2d_n(self,mode_num = 0,delta_n_fun = lambda x,y,e: 0,E2_norm = 1e3,iter = 10):
        self.mode_num = mode_num
        self.delta_n_fun = delta_n_fun
        self.E2_norm = E2_norm
        
        _, eig_vec = linalg.eigsh(self.Mat_s,k = self.mode_num + 1,which = "LA")
        self.field = eig_vec[:,-(1+mode_num)]
        
        ## iteration 1
        n1 = 2*iter//3 + 1
        amp_1 = np.linspace(0.2,1,n1)
        for i in range(n1):
            self._iteration_2d(amp_1[i])

        # iteration 2
        n2 = iter//3 + 1
        for _ in range(n2):
            self._iteration_2d(1.)
        
        eig_val, eig_vec = linalg.eigsh(self.Mat_i,k = self.mode_num + 1, which = "LA")
        
        Neff = np.sqrt(eig_val[-(1+mode_num)])/self.k0
        field = eig_vec[:,-(1+mode_num)]
        field_re = np.reshape(field,[self.N_mesh,self.N_mesh],"F")
        E2_ave = np.sum(np.abs(field)**2,axis = None)*self.hx*self.hy/(self.Lx*self.Ly)
        amp = np.sqrt(self.E2_norm/E2_ave)
        delta_n = self.delta_n_fun(self.xx,self.yy,field_re*amp) 
    
        self._visualization_2d(Neff,field_re*amp,delta_n+self.n_xy)
        
    def _iteration_2d(self,i_amp = 1.):
        E2_ave = np.sum(np.abs(self.field)**2,axis = None)*self.hx*self.hy/(self.Lx*self.Ly)
        amp = np.sqrt(self.E2_norm/E2_ave)*i_amp
        
        self.field_re = np.reshape(self.field,[self.N_mesh,self.N_mesh],"F") # reshape back to matric order
        delta_n = self.delta_n_fun(self.xx,self.yy,self.field_re*amp) 
        delta_n = np.reshape(delta_n,-1,"F") # reshape to natural order
        
        Mat_i_n = np.zeros_like(self.mat_2d) 
        Mat_i_n[np.arange(self.N_mesh**2),np.arange(self.N_mesh**2)] = ((delta_n + self.n_xy_reshape)*self.k0)**2
        
        self.Mat_i = self.mat_2d + Mat_i_n
        self.Mat_i = sparse.csr_matrix(self.Mat_i) # sparse
        
        _, eig_vec = linalg.eigsh(self.Mat_i,k = self.mode_num + 1,which = "LA")
        self.field = eig_vec[:,-(1+self.mode_num)]        
        
def main():
    x_bound = (-3*um,3*um)
    y_bound = (-4*um,4*um)
    
    a = 2*um
    b = 3*um
    c = 1.2*um
    d = 1*um
    
    n1 = 1.5
    n2 = 1.2
    
    # n_fun =  lambda x,y : n2 + (n1 - n2)*(np.abs(x) <= b/2)*(np.abs(y) <= a/2) + (n1-n2)*(np.abs(y-a/2-d/2)<=d/2)*(np.abs(x) <= c/2) # + (n1-n2)*(np.abs(y+a/2+d/2)<=d/2)*(np.abs(x) <= c/2)
    n_fun =  lambda x,y : n2 + (n1 - n2)*(np.abs(x) <= b/2)*(np.abs(y) <= a/2)
    # n_fun = lambda x,y : n2 + (n1-n2)*(np.sqrt(x**2 + y**2) <= a )
    
    lbd = 780*nm
    wvg = waveguide_2d(lbd,n_fun,x_bound = x_bound,y_bound = y_bound,N_mesh = 80)
    wvg.waveguide_2d(mode_num = 1)
    
def main_n():
    x_bound = (-3*um,3*um)
    y_bound = (-4*um,4*um)
    
    a = 2*um
    b = 3*um
    c = 1.2*um
    d = 1*um
    
    d2 = 1e-7
    E2_norm = 1e4
    
    n1 = 1.5
    n2 = 1.2
    
    # n_fun =  lambda x,y : n2 + (n1 - n2)*(np.abs(x) <= b/2)*(np.abs(y) <= a/2) + (n1-n2)*(np.abs(y-a/2-d/2)<=d/2)*(np.abs(x) <= c/2) # + (n1-n2)*(np.abs(y+a/2+d/2)<=d/2)*(np.abs(x) <= c/2)
    n_fun =  lambda x,y : n2 + (n1 - n2)*(np.abs(x) <= b/2)*(np.abs(y) <= a/2)
    # n_fun = lambda x,y : n2 + (n1-n2)*(np.sqrt(x**2 + y**2) <= a )
    
    delta_n_fun = lambda x,y,E : d2*np.abs(E)**2*(np.abs(x) <= b/2)*(np.abs(y) <= a/2)
    
    lbd = 780*nm
    wvg = nonlinear_waveguide_2d(lbd,n_fun,x_bound = x_bound,y_bound = y_bound,N_mesh = 80)
    wvg.waveguide_2d_n(mode_num = 1,delta_n_fun = delta_n_fun,E2_norm = E2_norm, iter = 6)
    
if __name__ == "__main__":
    # main()         
    main_n()
        
        