#%%
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from units import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, kv, jn_zeros # bessel function, hankel function, zeros of integer ordered bessel
from scipy.optimize import root_scalar
"""
calculation of fiber mode distribution and effective index N_eff (beta = k0*N_eff) from helmholtz equation -> fiber main equation: 

X*J_{l+1}(X)/( J_{l}(x) ) = Y*K_{l+1}(Y)/( K_{l}(Y) ); X**2 + Y**2 = V**2 = (k0*NA*a)**2

where a is radius of fiber core, k0 is vacuum wavevector, X = a* np.sqrt((n1*K0)**2 - beta**2) = kt*a , Y = a* np.sqrt(beta**2 - (n2*k0)**2) = gamma*a

U = { J_{l}(kt*r)
    { B*K_{l}(gamma*r)

"""
def fiber_plot(opc = "plot",**kwargs):

    lbd = kwargs.get("lbd",452.0*nm)
    n1 = kwargs.get("n1",1.4457) # core
    n2 = kwargs.get("n2",1.4378) # cladding
    a = kwargs.get("a",8*um)
    L = kwargs.get("L",0) # azimuthal index
    M = kwargs.get("M",2) # solution order of bessel function   

    V = np.sqrt(n1**2 - n2**2)*2*np.pi*a/lbd # V paramater
    k0 = 2*np.pi/lbd
    ## check whether we can solve this mode
    zeros_list = jn_zeros(L+1,M+1)
    #%%
    if M == 1:
        X_low = 0
    else:
        X_low = zeros_list[M-2]
    X_high = jn_zeros(L,M)[-1]
    
    if V < X_low:
        raise ValueError("can't guide that mode")
        
    if X_high >= V:
        X_high = V    
        
    ## solve LP_LM mode
    def func(X,V,L):
        Y = np.sqrt(V**2 - X**2)
        f = X*jv(L+1,X)/jv(L,X) - Y*kv(L+1,Y)/kv(L,Y)
        return f 

    x0 = 0.99*X_low+0.01*X_high
    x1 = 0.99*X_high+0.01*X_low
    
    Rootresult = root_scalar(func,args = (V,L),x0 = x0,x1 = x1,method = "brenth",bracket = (x0,x1),rtol = 1e-4)
    X_mode = Rootresult.root
    # X_mode = root_scalar(func,args = (V,L),bracket = (X_low,X_high),method = "bisect")

    kt = X_mode/a # kt = np.sqrt(n1**2 - n_e**2)*k0
    gamma = np.sqrt(V**2 - X_mode**2)/a # gamma
    N_eff = np.sqrt(n1**2 - (kt/k0)**2)

    if opc == "calculate":
        return N_eff
    elif opc == "plot":
        string = f"for LP_{L},{M} mode, N_eff = {N_eff:.10f}"
        print(string)

        ## figure
        x = np.linspace(-2.7*a,2.7*a,400)
        xx,yy = np.meshgrid(x,x)

        r = np.linspace(1e-10,2.7*a,200)
        rr = np.sqrt(xx**2 + yy**2)
        phi = np.arctan2(yy,xx)

        B = jv(L,kt*a)/kv(L,gamma*a)

        R = (r<=a)*jv(L,kt*r) + B*(r>a)*kv(L,gamma*r)
        RR = ( (rr<=a)*jv(L,kt*rr) + B*(rr>a)*kv(L,gamma*rr) )

        U_p = RR*np.exp(-1j*L*phi)
        U_n = RR*np.exp(1j*L*phi)

        fig, axes = plt.subplots(1,3)

        digital_r_out = 200/2.7
        a_in = L/kt
        digital_r_in = a_in/(2.7*a/200)

        ax0 = axes[0]
        im0 = ax0.imshow(np.abs((U_n + U_p)/np.sqrt(2))**2)
        ax0.set_title("Intensity distribution for U1")
        circle_out =  plt.Circle((200, 200), digital_r_out, color='white', fill=False, linewidth=1)
        circle_in =  plt.Circle((200, 200), digital_r_in, color='red', fill=False, linewidth=1)

        fig.colorbar(im0,ax = ax0)
        ax0.add_artist(circle_out)
        ax0.add_artist(circle_in)

        ax0.set_xlabel("x/grid")
        ax0.set_ylabel("y/gird")

        ax1 = axes[1]
        im1 = ax1.imshow(np.abs((U_n - U_p)/np.sqrt(2))**2)
        ax1.set_title("Intensity distribution for U2")
        circle_out =  plt.Circle((200, 200), digital_r_out, color='white', fill=False, linewidth=1)
        circle_in =  plt.Circle((200, 200), digital_r_in, color='red', fill=False, linewidth=1)

        fig.colorbar(im1,ax = ax1)
        ax1.add_artist(circle_out)
        ax1.add_artist(circle_in)

        ax1.set_xlabel("x/grid")
        ax1.set_ylabel("y/gird")

        ax2 = axes[2]
        ax2.plot(r*1e6,R,label = "R(r)")
        ax2.set_title("R(r) distribution" + string)
        ax2.set_xlabel("r/um")
        ax2.axvline(x=a*1e6, color='green', linestyle='--', linewidth=1.5,label = f"fiber core at r = a = {a*1e6:.2f}um")
        if L != 0:
            ax2.axvline(x=a_in*1e6, color='red', linestyle='--', linewidth=1.5,label = f"nearly no field inside r = {a_in*1e6:.2f}um")

        ax2.grid()
        ax2.legend()

        plt.figure(2)
        Xlist = np.linspace(0.01,V,400)
        Ylist = np.sqrt(V**2 - Xlist**2)
        plt.scatter(Xlist,Xlist*jv(L+1,Xlist)/jv(L,Xlist),s = 2,color = 'r',label = "plot - X")
        plt.plot(Xlist,Ylist*kv(L+1,Ylist)/kv(L,Ylist),label = "plot - Y")
        plt.ylim(-1,V+5)
        plt.grid()
        plt.legend()

        plt.show()
    else:
        pass

if __name__ == "__main__":
    fiber_plot("plot")




    


