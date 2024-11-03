import numpy as np
import matplotlib.pyplot as plt
from propagation_analytical import propagate_analytical
from units import *

dx = 1*um
Nx = 256; Ny = 256
w0 = 60*um
lbd = 589.0*nm
z0 = pi*w0**2/lbd
print(z0)

delta_x = 0
delta_y = 0
d = 0.3*cm

xlist = np.arange(-Nx//2,Nx//2)*dx
ylist = np.arange(-Ny//2,Ny//2)*dx

xx, yy = np.meshgrid(xlist,ylist[::-1])
rr = np.sqrt(xx**2 + yy**2)
theta = np.arctan2(yy-delta_y*dx,xx-delta_x*dx)

w1 = w0*np.sqrt(1+(d/z0)**2)

E_raw = np.exp(-(rr/w0)**2)
E_slm = E_raw*np.exp(-1j*theta)
E_slm_propagate = propagate_analytical(E_slm,lbd,d,dx)

E_raw_propagate = propagate_analytical(E_raw,lbd,d,dx)
R_g = d + z0**2/d
#E_gaussian_p = w0/w1*np.exp(-(rr/w1)**2)*np.exp(-1j*2*pi*d/lbd)*np.exp(-1j*pi/lbd*rr**2/R_g)*np.exp(1j*np.arctan(d/z0))

plt.subplot(1,2,1)
plt.imshow(np.abs(E_raw_propagate))
plt.colorbar()
plt.title("propagated gaussian beam")

plt.subplot(1,2,2)
plt.imshow(np.angle(E_slm_propagate),cmap = "rainbow")
plt.colorbar()
plt.title(f"sprial phase slm, with phase defect delta_x = {delta_x*dx/lbd:.2f}*lbd, delta_y = {delta_y*dx/lbd:.2f}*lbd")

plt.figure(2)
plt.plot(np.abs(E_slm_propagate[Ny//2,Nx//2::])**2)
plt.xlabel("x/um")
plt.title("radius distribution")

plt.show()
 

