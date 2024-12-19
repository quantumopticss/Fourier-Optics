import sys, os
path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(path))))

import numpy as np
from numpy import fft
from units import *
import matplotlib.pyplot as plt

class temporal_compression:
    def __init__(self,A_func,args = (),tlist = np.linspace(-1*ms,1*ms,1000)):
        """
        Args
        ---------
            A_func (callable): 
            args (tuple, optional): argumnets for A_func, Defaults to ().
            tlist (np.ndarray, optional): working time list, Defaults to np.linspace(-1*ms,1*ms,1000).
        """
        ## define A(t)
        self.tlist = tlist
        self.Alist = A_func(tlist,*args)
        
    def temporal_compression(self,phi_func,filter_freq_func,freq_list = np.arange(-500,500)/(2*ms),args = ((),(),)):
        """
        Args
        ---------
            phi_func (callable): phase added from modulator
            filter_freq_func (callable): transfer function in frequency domain, [-f,f]
            freq_list(np.ndarray, optional): working freq list matching the working time list, [-f, 0, f], defeault is np.arange(-500,500)/(2*ms)
            args (tuple, optional): arguments for the above two function, Defaults to ((),(),).
        """
        self.freq_list = freq_list
        self.f_Alist = fft.fftshift(fft.fft(self.Alist)) ## f(A)
        
        self.Alist_PM = self.Alist * np.exp( 1j*phi_func( self.tlist,*(args[0]) ) )
        self.f_Alist_PM = fft.fftshift(fft.fft(self.Alist_PM)) ## f(A_QPM )
        
        self.H_f_Alist = filter_freq_func(freq_list,*(args[1])) * self.f_Alist_PM ## f (H * A_QPM)        
        self.C_Alist = fft.ifft(self.H_f_Alist)
        
    def visualization(self):
        plt.figure(1)
        plt.plot(self.tlist*1e3, np.abs(self.Alist),label = "raw pulse")
        plt.plot(self.tlist*1e3, np.abs(self.C_Alist), label = "compressed pulse")
        plt.xlabel("time/ms")
        plt.ylabel("Amp")
        plt.title("pulse in time domain")
        plt.legend()
        
        plt.figure(2)
        plt.plot(self.freq_list, np.abs(self.f_Alist),label = "F of raw pulse")
        plt.plot(self.freq_list, np.abs(self.H_f_Alist), label = "F of pulse_PM")
        plt.plot(self.freq_list, np.abs(self.H_f_Alist), label = "F of compressed pulse")
        plt.xlabel("freq/Hz")
        plt.ylabel("Amp")
        plt.title("pulse in frequency domain")
        plt.legend()
        
        plt.show()

def main():
    tau = 0.75*ms
    t0 = 0*ms
    zeta = 1.2/(ms**2) # 
    tlist = np.linspace(-5*ms,5*ms,2000)
    f_list = np.arange(-1000,1000)/(10*ms)
    
    A_func = lambda t : np.exp(-((t-t0)/tau)**2)

    phase_func = lambda t : zeta*(t-t0)**2
    
    ##
    a = zeta*tau**2
    b = -tau**2*a
    phis = lambda f : 2*pi*2*ms*f + b*pi**2*f**2 # delay = 0.05*ms, b = 0.2
    H_trans = lambda f : np.exp(-1j*phis(f)) 
    
    tau_2 = tau/np.sqrt(1+a**2)
    print(tau_2*1e3)
    
    eg = temporal_compression(A_func,tlist = tlist)
    eg.temporal_compression(phase_func,H_trans,freq_list=f_list)
    eg.visualization()
    
if __name__ == "__main__":
    main()
        