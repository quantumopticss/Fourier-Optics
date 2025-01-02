import sys, os
path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(path))))

import numpy as np
import odesolver as ode
from scipy.integrate import solve_ivp
from units import *
import matplotlib.pyplot as plt
from scipy.special import factorial
from matplotlib.animation import FuncAnimation

class SVE_diffusion:
    # vector_factorial = np.vectorize(factorial) # vectorize a normal function to support ndarray input
    def __init__(self,vg = c0_const, D_nu = None , d_beta_args = [],scaling = (1*ps, 1*mm)):
        """ slowly varying envelop(SVE) diffusion equation for ultrafast pulse evolution 
        
        Args
        ---------
            vg : float
                group velocity
                
            D_nu : float
                group velocity dispersion(GVD) , defeault is None - No GVD
                
            d_beta : np.ndarray
                higher order derivatives of beta - omega, defeault is []
                
            scaling : tuple
                scaling indecies to match you time and space scale,
                defeault is (1*ps, 1*mm)
        """  
        
        
        if d_beta_args != []:
            if d_beta_args[-1] == 0:
                raise ValueError("the latest d_beta_args should not be zero")
        
        self.t0, self.a0 = scaling # scaling indecies
        self.vg = vg
        if D_nu != None:
            self.num =  2 + len(d_beta_args) # order of derivative function, basic include D_nu
        else:
            self.num =  1 + len(d_beta_args) # order of derivative function, D_nu in extra_args
        
        i_list = np.arange(self.num) + 1
        self.factor = ((-1j/self.t0)**i_list) / factorial(i_list)
        
        ## p_t list
        if D_nu != None:
            f = np.array(([1/vg,D_nu/(2*pi)] + [i for i in d_beta_args]),dtype = np.clongdouble) # dn_nu_beta
        else:
            f = np.array(([1/vg] + [i for i in d_beta_args]),dtype = np.clongdouble) # dn_nu_beta
            
        f = f*self.factor
        f_sum = f[::-1]
        inv = (1. / f_sum[-1])
        
        
        def dadt(t,A):
            ## A = [A,pt_A,2pt_A,...(num-1)pt_A], in (:,i)
            ## -1j*pz_A + sum_{n=1}^{num-1} dn_beta * (-1j)**n / n! n_pt_A + d_num_beta * (-1j)**num/ num! num_pt_A = 0
            
            
            ######## valid for normal ode form
            # pz_A = (( np.roll(A[:,0],1) - np.roll(A[:,0],-1) )/(2*self.dz)) 
            # dAdt = np.roll(np.copy(A),-1,axis = 1)

            # dAdt[:,-1] = ( 1j*pz_A - np.sum( ( f_sum[np.newaxis,:] * dAdt[:,::-1] ) ,axis = 1) ) * inv 
            # dAdt[:,-1] = -1j*4*pi/D_nu * (AA[:,1]/vg + pz_A) # second order
            
            ######## valid for normal ode form
            AA = np.reshape(A,[self.length,-1],order = "F")
            pz_A = (( np.roll(AA[:,0],1) - np.roll(AA[:,0],-1) )/(2*self.dz)) 
            dAdt = np.roll(np.copy(AA),-1,axis = 1)
            
            dAdt[:,-1] = ( 1j*pz_A - np.sum( ( f_sum[np.newaxis,:] * dAdt[:,::-1] ) ,axis = 1) ) * inv 
            # dAdt[:,-1] = -1j*4*pi/D_nu * (AA[:,1]/vg + pz_A) # second order
            dAdt = np.reshape(dAdt,newshape = -1,order = "F") #### reshape for solivp
            
            print(t) # time prob
            
            return dAdt
        self.func = dadt
                
    def sve_diffusion(self,Afunc_z0,z_tuple = ([],1*pm),args = (),t_tuple = ([0.,10*ps], 0.01*ps)):
                
        ## parameters
        self.zlist, self.dz = z_tuple
        tspan , dt = t_tuple
        
        self.length = len(self.zlist)
        
        # add scaling
        tspan = np.array(tspan)/self.t0
        dt /= self.t0
        
        ## initial conditions
        self.A0 = Afunc_z0(self.zlist,*args) # A(z,0)
        self.initial = np.zeros([len(self.A0),self.num],dtype = np.clongdouble) # 0 ~ num-1
        self.initial[:,0] = self.A0
        
        # if self.num >= 2:
        #     self.initial[:,1] = - self.vg * ( np.roll(self.A0,1) - np.roll(self.A0,-1) )/(2*self.dz)
        
        ## fixed step ode
        
        # self.tlist , self.Alist = ode.ode23(self.func,tspan[0],self.initial,tspan[1], step_max = dt)
        # self.tlist *= self.t0 # scaling
        
        self.initial = np.reshape(self.initial,newshape=-1,order = "F")
        t_eval = np.arange(tspan[0],tspan[1],dt)
        solution = solve_ivp(self.func, tspan, self.initial, method='BDF', t_eval=t_eval,vectorized=True)
        
        self.tlist = solution.t * self.t0
        self.Alist = np.reshape(solution.y,newshape=[len(self.tlist),-1],order = "F")
        
    def visualization(self):
        fig, ax = plt.subplots()

        # Set up the plot limits
        y_max = np.max(np.abs(self.Alist[:,:,0]), axis=None)
        ax.set_ylim(-0.15 * y_max, y_max * 1.2)  # y-axis limits

        # Initial plot setup
        initial_line, = ax.plot(self.zlist*1e3, np.abs(self.Alist[0, :, 0]), label="initial")
        time_line, = ax.plot([], [], label="time")
        ax.set_xlabel("z/mm")
        ax.set_ylabel("t - amp")
        ax.legend()

        # Animation function: this will be called at each frame
        def animate(frame):
            time_line.set_data(self.zlist*1e3, np.abs(self.Alist[frame, :, 0]))
            try:
                ax.set_title(f"time domain, t = {self.tlist[frame] * 1e12:.3f}ps")
            except:
                ax.set_title(f"time domain, t = {self.tlist[frame,0] * 1e12:.3f}ps")
            return time_line,

        # Create the animation object
        ani = FuncAnimation(fig, animate, frames=np.size(self.tlist), blit=True, interval=50)

        # Save the animation
        ani.save('SVE.gif', writer='Pillow', fps=30)
        print("end")
        # Show the animation
        plt.show()

def main():
    # 1ps -> 0.3mm
    tspan = [0.,6*ps]
    zlist,dz = np.linspace(-1*mm,4*mm,400,retstep=True)
    tau = 0.4*ps
    step = 0.006*ps
    
    v = 0.8*c0_const
    D_nu = 1.82*1e-24
    
    z0 = 1*mm # pulse start at z0
    
    a = 1e-2 # chirp parameter
    Afunc_z0 = lambda z,z0: np.exp(- (((z-z0)/(v))**2 * (1 - 1j*a)/tau**2) )
    
    sve_diffu = SVE_diffusion(v)
    sve_diffu.sve_diffusion(Afunc_z0,(zlist,dz,),args = (z0,),t_tuple = (tspan,step,))
    sve_diffu.visualization()
    
if __name__ == "__main__":
    main()