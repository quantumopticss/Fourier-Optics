import numpy as np
import matplotlib.pyplot as plt
from units import *

# phase = k*r - w*t

def _propagation_analytical_transfer(E_input,lbd,d,dx,mask = 1.0):
    # transfer function H(nu_x,nu_y) = np.exp(-1j*2*pi*d*np.sqty((1/lbd)**2 - nu_x**2 - nu_y**2))
    shape_E = np.shape(E_input)
    dimension = len(shape_E)
    if dimension == 2:
        N, M = E_input.shape
        
        nxlist = np.arange(-M//2,M//2)
        nylist = np.arange(-N//2,N//2)
        
        xx, yy = np.meshgrid(nxlist,nylist)
        
        nu_xx = xx/(M*dx)
        nu_yy = yy/(N*dx)
        
        H = np.exp(-1j*2*np.pi*d*np.sqrt((1/lbd)**2-nu_xx**2-nu_yy**2))   # analytical transfer function
        
        f_E = np.fft.fftshift(np.fft.fft2(E_input*mask),axes = (0,1))
        E_propagate = np.fft.ifft2(f_E*H)
        
    if dimension == 1:
        N = np.size(E_input)
        x = np.arange(-N//2,N//2)
        
        nu_x = x/(N*dx)
        
        H = np.exp(-1j*2*pi*d*np.sqrt((1/lbd)**2 - nu_x**2))
        f_E = np.fft.fftshift(np.fft.fft(E_input*mask))
        
        E_propagate = np.fft.ifft(H*f_E)
    
    return E_propagate
    
def _propagation_analytical_impulse(E_input,lbd,d,dx,mask = 1.0):
    shape_E = np.shape(E_input)
    dimension = len(shape_E)
    if dimension == 2:
        N, M = E_input.shape
        
        nxlist = np.arange(-M//2,M//2)*dx
        nylist = np.arange(-N//2,N//2)*dx
        
        xx, yy = np.meshgrid(nxlist,nylist)
        rr = np.sqrt(xx**2 + yy**2 + d**2)
        
        h = d/(rr**2) * (1/rr + 1j*2*pi/lbd)*np.exp(-1j*2*pi*rr/lbd)/(2*pi)  # analytical impulse response function
        
        H = np.fft.fft2(h)   
        f_E = np.fft.fft2(E_input*mask)
        
        E_propagate = np.fft.ifftshift(np.fft.ifft2(f_E*H),axes = (0,1))*(dx**2)
    if dimension == 1:
        1
        
    return E_propagate
    
def propagate_analytical(E_input,lbd,d,dx,mask = 1.0):
    n = len(np.shape(E_input))
    if n == 2:
        N = np.sqrt(np.size(E_input))
    elif n == 1:
        N = np.size(E_input)
    
    if (dx**2) > lbd*d/(N):
        print("transfer")
        E_propagate = _propagation_analytical_transfer(E_input,lbd,d,dx,mask)
    else:
        print("impulse")
        E_propagate = _propagation_analytical_impulse(E_input,lbd,d,dx,mask)
    
    return E_propagate
    
def test_1d():
    d1 = 0.2*m
    dx = 0.05*mm

    z0 = 5*m
    w0 = 1*mm
    lbd = pi*w0**2/z0
    k = 2*pi/lbd
    R1 = -(d1 + z0**2/d1)
    print(lbd*1e9)

    x = np.arange(-100,100)
    w1 = w0*np.sqrt(1+(d1/z0)**2)
    E = (x*dx/w0)*np.exp(-(x*dx/w1)**2)*np.exp(-1j*k*(x*dx)**2/(2*R1))
    
    E_simu = propagate_analytical(E,lbd,2*d1,dx)
    E_ana = (x*dx/w0)*np.exp(-(x*dx/w1)**2)*np.exp(-1j*k*(x*dx)**2/(-2*R1))
    
    plt.figure(1)
    plt.plot(x,np.abs(E_simu),label = "simu")
    plt.plot(x,np.abs(E_ana),label = "ana")
    plt.plot(x,np.abs(E),label = "raw")
    plt.legend()
    plt.show()
    
def test_2d():
    d1 = 2*m
    dx = 0.09*mm

    z0 = 5*m
    w0 = 1*mm
    lbd = pi*w0**2/z0
    k = 2*pi/lbd
    R1 = -(d1 + z0**2/d1)
    print(lbd*1e9)

    x = np.arange(-100,100)
    y = np.arange(-100,100)
    xx,yy = np.meshgrid(x,y)
    
    rr = np.sqrt(xx**2 + yy**2)
    
    w1 = w0*np.sqrt(1+(d1/z0)**2)
    
    E = xx*yy*w0/w1*np.exp(-(rr*dx/w1)**2)*np.exp(-1j*k*(rr*dx)**2/(2*R1))*np.exp(1j*3*np.arctan(d1/z0))
    
    E_simu = propagate_analytical(E,lbd,2*d1,dx)
    E_ana = xx*yy*w0/w1*np.exp(-(rr*dx/w1)**2)*np.exp(-1j*k*(rr*dx)**2/(-2*R1))*np.exp(-1j*3*np.arctan(d1/z0))
    
    plt.figure(1)
    plt.imshow(np.abs(E_simu),label = "simu")
    plt.title("simu")
    plt.colorbar()
    
    plt.figure(2)
    plt.imshow(np.abs(E_ana),label = "ana")
    plt.title("ana")
    plt.colorbar()
    
    plt.figure(3)
    plt.imshow(np.abs(E),label = "raw")
    plt.title("raw")
    plt.colorbar()
    
    plt.show()
 
if __name__ == "__main__":
    #test_1d()
    test_2d()