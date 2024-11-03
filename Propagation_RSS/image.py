import numpy as np
import matplotlib.pyplot as plt
from units import *
from propagation_fresnel import propagate

path = "fig.png"
I_fig = plt.imread(path) 
E_image = np.empty_like(I_fig,dtype = complex)
for i in range(3):
    I_fig_sub = I_fig[:,:,i]
    N, M = np.shape(I_fig)[0], np.shape(I_fig)[1]
    lbd = 50.3*nm
    dx = 10*um

    E_obj = np.zeros([3*N,3*M])
    E_obj[N:2*N,M:2*M] = np.sqrt(I_fig_sub) 

    d1 = 4*m
    d2 = 4*m
    f = 1/(1/d1+1/d2)

    x = np.arange(-3*M//2,3*M//2)
    y = np.arange(-3*N//2,3*N//2)

    xx,yy = np.meshgrid(x,y)

    rR = np.sqrt(xx**2 + yy**2)*dx

    a = 0.4*cm

    phase_f = pi*rR**2/(lbd*f)
    mask_f = np.exp(1j*phase_f)*(rR<=a)

    E_f1 = propagate(E_obj,lbd,d1,dx)
    E_image[:,:,i] = propagate(E_f1,lbd,d2,dx,mask_f)[2*N:N:-1,2*M:M:-1]
    
I_image = np.abs(E_image)**2
I_image[:,:,-1] = I_fig[:,:,-1]
 
plt.figure(1)
plt.imshow(I_fig)
plt.colorbar()
plt.title("raw")

plt.figure(2)
plt.imshow(I_image)
plt.colorbar()
plt.title("image")

plt.show()
