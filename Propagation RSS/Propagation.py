import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
from ..units import *

# phase = k*r - w*t

def propagation(E_field,delta_x = 1*um,delta_y = 1*um,lbd = 589.3*nm,z = 10*cm):
    """
    Propagation under fresnel approximation
    """
    
    N,M = np.shape(E_field)[0], np.shape(E_field)[1]
    # reshape E_field
    if N%2 == 1:
        E_field = E_field[0:N,...]
        N -= 1
    if M%2 == 1:
        E_field = E_field[:,0:M]
        M -= 1
     
    # meshgrid
    Nx_list = np.arange(-M//2,M//2)
    Ny_list = np.arange(-N//2,N//2)
    
    judge_x = delta_x**2*M/(lbd*z)
    judge_y = delta_y**2*N/(lbd*z)
    print(judge_y,judge_x)
    
    Nf = (np.max([N,M])*np.max([delta_x,delta_y]))**4*4/(z**3*lbd)
    if Nf >= 1:
        print("warning! Fresnel condition is not well satisfied") 
        
    ## fresnel propagation over x axis
    if judge_x >= 1: # over sampled for transfer function
        fx_E = fft.fftshift(fft.fft(E_field,axis = 1),axes = 1)
        fx_list = Nx_list/(M*delta_x)
        H_x = np.exp(-1j*pi*lbd*z*fx_list**2) 
        H_x = H_x[np.newaxis,:] # Hx operate on axis 1 
        E_propagate_x = fft.ifft(fx_E*H_x,axis = 1)
        
    else: # over sampled for impulse response function 
        # h_x = exp(1j*pi*(x-x')**2/(lbd*z))
        
        M_fresnel_x = np.zeros([M,M])
        for i in range(M):
            M_fresnel_x[:,i] = - i + np.arange(M)
            
        M_fresnel_x = np.exp(1j*pi*(M_fresnel_x*delta_x)**2/(lbd*z)) 
        E_propagate_x = E_field @ M_fresnel_x * delta_x * np.sqrt(-1j/(lbd*z))
    
    ## fresnel propagation over y axis
    if judge_y >= 1: # over sampled for transfer function
        fy_E = fft.fftshift(fft.fft(E_propagate_x,axis = 0),axes = 0)
        fy_list = Ny_list/(N*delta_y)

        H_y = np.exp(-1j*pi*lbd*z*fy_list**2)
        H_y = H_y[:,np.newaxis] # Hy operate on axis 0 
        E_propagate_xy = fft.ifft(fy_E*H_y,axis = 0)
        
    else: # over sampled for impulse response function
        M_fresnel_y = np.zeros([N,N])
        for i in range(N):
            M_fresnel_y[i,:] = - i + np.arange(N)
            
        M_fresnel_y = np.exp(1j*pi*(M_fresnel_y*delta_y)**2/(lbd*z)) 
        E_propagate_xy =  M_fresnel_y @ E_propagate_x * delta_y * np.sqrt(-1j/(lbd*z))
    
    return E_propagate_xy
    
def propagation_test():
    E_in = np.zeros([200,200])
    E_in[80:121,80:121] = 0.9
    
    delta = 10*um
    lbd = 500*nm
    z = 30*cm

    E_propagated = propagation(E_in,delta_x=delta,delta_y=delta,lbd=lbd,z=z)
    
    plt.subplot(1,2,1)
    plt.imshow(np.abs(E_propagated)**2)
    plt.colorbar()
    
    plt.subplot(1,2,2)
    plt.plot((np.abs(E_propagated)**2)[100,:])
    
    plt.show()
 
if __name__ == "__main__":
    propagation_test()