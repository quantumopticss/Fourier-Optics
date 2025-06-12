#%%
import os,sys
units_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(units_path)
#%%
import numpy as np
import matplotlib.pyplot as plt
from units import *

def test_1d():
    d1 = 2*m
    dx = 0.01*mm

    w0 = 2*mm
    lbd = 780*nm
    z0 = pi*w0**2/lbd

    k = 2*pi/lbd
    R1 = -(d1 + z0**2/d1)
    print(z0)

    x = np.arange(-100,100)
    w1 = w0*np.sqrt(1+(d1/z0)**2)
    E = (x*dx/w0)*np.exp(-(x*dx/w1)**2)*np.exp(-1j*k*(x*dx)**2/(2*R1))*np.exp((1+0.5)*1j*np.arctan(d1/z0))
    
    E_simu = propagate(E,lbd,2*d1,dx)
    E_ana = (x*dx/w0)*np.exp(-(x*dx/w1)**2)*np.exp(-1j*k*(x*dx)**2/(-2*R1))*np.exp(-1.5*1j*np.arctan(d1/z0))
    
    plt.figure(1)
    plt.plot(x,np.real(E_simu),label = "simu")
    plt.plot(x,np.real(E_ana),label = "ana")
    plt.plot(x,np.real(E),label = "raw")
    plt.legend()
    plt.show()
    
def f_test():
    d = 30*cm
    dx = 0.04*mm
    lbd = 780*nm
    
    E:np.ndarray = np.load("data.txt.npy")
    # E = (E >= 0.8)
    N,M = E.shape
    xx:np.ndarray = np.load("xx.txt.npy")
    yy = xx.copy().T
    E_simu = propagate(E,lbd,d,dx,method = None)
    
    plt.subplot(2,2,1)
    plt.imshow(np.abs(E)/np.abs(E).max(),label = "raw")
    plt.title("raw")
    plt.colorbar()
    
    plt.subplot(2,2,2)
    plt.imshow(np.abs(E_simu),label = "simu")
    plt.title("simu")
    plt.colorbar()
    
    plt.subplot(2,2,3)
    plt.plot(np.abs(E_simu)[N//2,:]/np.abs(E).max(),label = "raw_cut")
    plt.legend()
    
    plt.subplot(2,2,4)
    plt.plot(np.abs(E)[N//2,:],label = "pro_cut")
    plt.legend()
    
    plt.show()

def test_2d():
    d1 = 40*cm
    dx = 0.03*mm

    lbd = 780*nm
    w0 = 3*mm
    
    z0 = pi*w0**2/lbd
    k = 2*pi/lbd
    R1 = -(d1 + z0**2/d1)
    print(lbd*1e9)

    x = np.arange(-800,800)
    y = np.arange(-800,800)
    xx,yy = np.meshgrid(x,y)
    
    rr = np.sqrt(xx**2 + yy**2)
    
    w1 = w0*np.sqrt(1+(d1/z0)**2)
    
    E = w0/w1*np.exp(-(rr*dx/w1)**2)*np.exp(-1j*k*(rr*dx)**2/(2*R1))*np.exp(1j*np.arctan(d1/z0))
    
    E_simu = propagate(E,lbd,2*d1,dx,method = "impulse")
    E_ana = w0/w1*np.exp(-(rr*dx/w1)**2)*np.exp(-1j*k*(rr*dx)**2/(-2*R1))*np.exp(-1j*np.arctan(d1/z0))
    
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

def _propagation_impulse(E_input,lbd,d,dx,mask = 1.0):
    """_Propagation under fresnel approximation: impulse response function @ convolution
    E(x,y) = 1j*np.exp(-1j*k*d)/(d*lbd) * iint  U(x',y')*np.exp(-1j*pi/(lbd*d)*((x-x')**2 + (y-y')**2)) dx'dy'_
    """
    shape_E = np.shape(E_input)
    dimension = len(shape_E)
    if dimension == 2:
        N, M = E_input.shape
        
        M_i = np.empty([N,N])
        M_j = np.empty([M,M])
        
        for i in range(N):
            M_i[i,:] = -i + np.arange(N)
        for j in range(M):
            M_j[:,j] = -j + np.arange(M)
        
        M_i = np.exp(-1j*np.pi*(M_i*dx)**2/(d*lbd))
        M_j = np.exp(-1j*np.pi*(M_j*dx)**2/(d*lbd))

        E_propagate = 1j/(lbd*d)*np.exp(-1j*2*np.pi*d/lbd)*(dx**2) * (M_i @ (E_input*mask) @ M_j)
        
        """
        # or we can use :
        
        x = np.arange(-M//2,M//2)
        y = np.arange(-N//2,N//2)
        
        xx,yy = np.meshgrid(x,y)
        
        h = np.exp(-1j*pi*(xx**2+yy**2)*dx**2/(d*lbd))
        f_h = np.fft.fft2(h)
        f_E = np.fft.fft2(E_input*mask)
        E_propagate = 1j*np.exp(-1j*2*pi*d/lbd)/(lbd*d)*(dx**2) * np.fft.ifftshift(np.fft.ifft2(f_E*f_h),axes = (0,1))
        """
                
    elif dimension == 1:
        N = np.size(E_input)
        M_i = np.empty([N,N])
        for i in range(N):
            M_i[i,:] = -i + np.arange(N)
        
        M_i = np.exp(-1j*np.pi*(M_i*dx)**2/(d*lbd))
        E_propagate = np.sqrt(1j/(d*lbd))*np.exp(-1j*2*np.pi*d/lbd)* (M_i @ (E_input*mask))*dx
        
        """
        # or we can use :
        
        x = np.arange(-N//2,N//2)
        
        h = np.exp(-1j*pi*(x*dx)**2/(d*lbd))
        f_h = np.fft.fft(h)
        f_E = np.fft.fft(E_input*mask)
        E_propagate = np.exp(-1j*2*pi*d/lbd)*np.sqrt(1j/(lbd*d))*dx * np.fft.ifftshift(np.fft.ifft(f_E*f_h))
        """
        
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
        
        H = np.exp(1j*np.pi*d*lbd*(nu_xx**2+nu_yy**2))   # transfer function
        
        f_E = np.fft.fftshift(np.fft.fft2(E_input*mask),axes = (0,1))
        E_propagate = np.fft.ifft2(f_E*H)*np.exp(-1j*d*2*np.pi/lbd)*1j*dx**2/(d*lbd)
        
    if dimension == 1:
        N = np.size(E_input)
        xx = np.arange(-N//2,N//2)
        
        nu_xx = xx/(N*dx)
        H = np.exp(1j*np.pi*d*lbd*nu_xx**2)
        
        f_E = np.fft.fftshift(np.fft.fft(E_input*mask))
        E_propagate = np.fft.ifft(f_E*H)*np.exp(-1j*d*2*np.pi/lbd)*np.sqrt(1j/(d*lbd))*dx
    
    return E_propagate

def _propagation_fresnel(E_input,lbd,d,dx,mask = 1.0):
    """ g(x,y) = iint 1j/(d*lbd) * np.exp(-1j*2*pi*d/lbd) f(x',y') * np.exp(-1j*pi*((x-x')**2+(y-y')**2)/(lbd*d)) dx'dy'
               = 1j/(d*lbd) * np.exp(-1j*2*pi*d/lbd) * np.exp(-1j*pi*(x**2+y**2)/(d*lbd)) * iint f(x',y')*np.exp(-1j*pi*(x'**2+y'**2)/(d*lbd)) * np.exp(
                   2*pi*(x*x' + y*y')/(d*lbd)
               )
    """
    ...

def propagate(E_input,lbd,d,dx,mask=1.,*,method = None):
    N = np.sqrt(np.size(E_input))
    
    if method == "transfer":
        print("transfer")
        E_propagate = _propagation_transfer(E_input,lbd,d,dx,mask)        
        
    elif method == "impulse":
        print("impulse")
        E_propagate = _propagation_impulse(E_input,lbd,d,dx,mask)
    else:
        if (dx**2) > lbd*d/(N):
            print("transfer")
            E_propagate = _propagation_transfer(E_input,lbd,d,dx,mask)
        else:
            print("impulse")
            E_propagate = _propagation_impulse(E_input,lbd,d,dx,mask)
        
    
    return E_propagate

if __name__ == "__main__":
    # test_2d()
    f_test()
    #test_2d()