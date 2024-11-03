import numpy as np
import matplotlib.pyplot as plt
from units import *

def _test():
    d1 = 2*m
    dx = 0.1*mm

    z0 = 5*m
    w0 = 1*mm
    lbd = pi*w0**2/z0
    k = 2*pi/lbd
    R1 = -(d1 + z0**2/d1)
    print(lbd*1e9)

    x = np.arange(-100,100)
    w1 = w0*np.sqrt(1+(d1/z0)**2)
    E = (x*dx/w0)*np.exp(-(x*dx/w1)**2)*np.exp(-1j*k*(x*dx)**2/(2*R1))
    
    E_simu = propagate(E,lbd,2*d1,dx)
    E_ana = (x*dx/w0)*np.exp(-(x*dx/w1)**2)*np.exp(-1j*k*(x*dx)**2/(-2*R1))
    
    plt.figure(1)
    plt.plot(x,np.abs(E_simu)+0.2,label = "simu")
    plt.plot(x,np.abs(E_ana)+0.1,label = "ana")
    plt.plot(x,np.abs(E),label = "raw")
    plt.legend()
    plt.show()

def _propagation_impulse(E_input,lbd,d,dx,mask = 1.0):
    """_Propagation under fresnel approximation: impulse response function @ convolution
    E(x,y) = 1j*np.exp(-1j*k*d)/(d*lbd) * iint  U(x',y')*np.exp(-1j*pi/(lbd*d)*((x-x')**2 + (y-y')**2)) dx'dy'_
    """
    shape_E = np.shape(E_input)
    dimension = len(shape_E)
    if dimension == 2:
        N = shape_E[0]
        M = shape_E[1]
        
        M_i = np.empty([N,N])
        M_j = np.empty([M,M])
        
        for i in range(N):
            M_i[i,:] = -i + np.arange(N)
        for j in range(M):
            M_j[:,j] = -j + np.arange(M)
        
        M_i = np.exp(-1j*np.pi*(M_i*dx)**2/(d*lbd))
        M_j = np.exp(-1j*np.pi*(M_j*dx)**2/(d*lbd))

        E_propagate = 1j/(lbd*d)*np.exp(-1j*2*np.pi*d/lbd)*(dx**2) * (M_i @ (E_input*mask) @ M_j)
                
    elif dimension == 1:
        N = np.size(E_input)
        M_i = np.empty([N,N])
        for i in range(N):
            M_i[i,:] = -i + np.arange(N)
        
        M_i = np.exp(-1j*np.pi*(M_i*dx)**2/(d*lbd))
        E_propagate = np.sqrt(1j/(d*lbd))*np.exp(-1j*2*np.pi*d/lbd)* (M_i @ (E_input*mask))*dx
        
    return E_propagate

def _propagation_transfer(E_input,lbd,d,dx,mask = 1.):
    """_Propagation under fresnel approximation: transfer function
    E(x,y) = 1j*np.exp(-1j*k*d)/(d*lbd) * iint  U(x',y')*np.exp(-1j*pi/(lbd*d)*((x-x')**2 + (y-y')**2)) dx'dy'_
    """
    shape_E = np.shape(E_input)
    dimension = len(shape_E)
    if dimension == 2:
        N = shape_E[0]
        M = shape_E[1]
        
        nxlist = np.arange(-M//2,M//2)
        nylist = np.arange(-N//2,N//2)
        
        xx, yy = np.meshgrid(nxlist,nylist)
        
        nu_xx = xx/(M*dx)
        nu_yy = yy/(N*dx)
        
        H = np.exp(-1j*d*2*np.pi/lbd)*np.exp(1j*np.pi*d*lbd*(nu_xx**2+nu_yy**2))   # transfer function
        
        f_E = np.fft.fftshift(np.fft.fft2(E_input*mask),axes = (0,1))
        E_propagate = np.fft.ifft2(f_E*H)
        
    if dimension == 1:
        N = np.size(E_input)
        xx = np.arange(-N//2,N//2)
        
        nu_xx = xx/(N*dx)
        H = np.exp(-1j*2*np.pi*d/lbd)*np.exp(1j*np.pi*d*lbd*nu_xx**2)
        
        f_E = np.fft.fftshift(np.fft.fft(E_input*mask))
        E_propagate = np.fft.ifft(f_E*H)
    
    return E_propagate
            
def propagate(E_input,lbd,d,dx,mask=1.):
    N = np.sqrt(np.size(E_input))
    
    if (dx**2) > lbd*d/(N):
        print("transfer")
        E_propagate = _propagation_transfer(E_input,lbd,d,dx,mask)
    else:
        print("impulse")
        E_propagate = _propagation_impulse(E_input,lbd,d,dx,mask)
    
    return E_propagate

if __name__ == "__main__":
    _test()