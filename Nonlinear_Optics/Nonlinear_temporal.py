import sys,os
path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(path)))

import numpy as np
from units import *
from odesolver import odeint
import matplotlib.pyplot as plt

## 
class nonlinear_model:
    def __init__(self,omega_0,beta,a,b,F_func,args = ()):
        # define nonlinear lorentz mode
        # x'' + 2*beta*x' + w0**2*x + w0**2 * (a*x**2 + b*x**3) = F
        # dx/dt = x'
        # dx'/dt = x'' = F - 2*beta*x' - w0**2 * (x + a*x**2 + b*x**3 )

        def func(t,y):
            x, vx = y[0], y[1]
            try:
                ax = F_func(t,*args) - 2*beta*vx - omega_0**2 * (x + a*x**2 + b*x**3)
            except:
                ax = F_func - 2*beta*vx - omega_0**2 * (x + a*x**2 + b*x**3)
            dydt = np.array([vx,ax])
            return dydt
        
        self.ode_func = func
    
    def nonlinear_run(self,tspan,y_init,t_step = 1e-2,step_max = 5e-3):
        self.t_step = t_step
        step_max = np.min([t_step/2, step_max])
        self.tlist, self.ylist = odeint(self.ode_func,tspan[0],y_init,tspan[1],t_step = t_step,step_max = step_max)
        return self

    def visual(self):
        N = np.size(self.tlist) # N
        T_range = self.tlist[-1] - self.tlist[0]
        
        self.f_list = np.arange(-N//2,N//2)/T_range
        self.f_x = np.fft.fftshift(np.fft.fft(self.ylist[:,0]))/N

        ## fig 1
        fig1, ax1 = plt.subplots()
        ax1.plot(self.f_list, np.abs(self.f_x),label = "fft of x")
        ax1.set_xlabel("freq/Hz")
        ax1.set_ylabel("spectrum amp")
        ax1.set_title("freq domain")

        ## fig 2
        fig2, ax2 = plt.subplots()
        ax2.plot(self.tlist,self.ylist[:,0],label = "x")

        ax2_v = ax2.twinx()
        ax2_v.plot(self.tlist,self.ylist[:,1],label = "v",c = "r")

        ax2.set_xlabel("time/s")
        ax2.set_ylabel("x")
        ax2_v.set_ylabel("velocity")
        ax2.set_title("time domain")

        # Combine the handles and labels from both axes
        handles, labels = [], []
        for ax in [ax2, ax2_v]:
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)

        # Display the combined legend
        fig2.legend(handles, labels)

        plt.show()


def main():
    w = 20
    T = 5
    t_step = 0.005
    beta = 1.5

    a = 0
    b = 60

    f_func = lambda t: float(60*np.cos(w*2*np.pi*t))
    nonlinear = nonlinear_model(w,beta,a,b,f_func)
    nonlinear.nonlinear_run([0.,T],[0.,0.],t_step).visual()

if __name__ == "__main__":
    main()


