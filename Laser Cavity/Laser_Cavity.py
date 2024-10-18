"""
Finite size of reflector -> 

2D gaussian beam

differection loss & change field distribution 

"""
#%% imports
from units import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_hermite
from propagation_fresnel import propagate
#%% parameters
a = 0.7*cm
d = 1*m
R1 = -1.5*m
R2 = -1.5*m

# from spherical mirror cavity we know
z0 = np.sqrt(-(R1+d)*(R2+d)*(R1+R2+d)*d/((R1+R2+2*d)**2))
print("z0 = ",z0,"m")

d1 = d*(R2+d)/(R1+R2+2*d)
d2 = (R1+d)*d/(R1+R2+2*d)

n_L = 4200000 # longitudinal number
n_T = 0 # transverse number

freq = c0_const/(2*d) * (n_L + (n_T+0.5)*(np.arctan(d1/z0) + np.arctan(d2/z0))/pi) 
lbd = c0_const/freq
print("lbd = ", lbd*1e9 ,"nm")

w0 = np.sqrt(z0*lbd/pi)
k = 2*pi*lbd
#%% calculations
# U2(x,y) = 1j*np.exp(-1j*k*d)/(lbd*d) * iint U1(x',y')*np.exp(   -1j*pi/lbd*(  (x'**2 + y'**2)/R1 +  (x**2 + y**2)/R2  )  )
#                                                      *np.exp(-1j*pi/d*((x-x')**2) + (y-y')**2) dx'dy'

# operate
N = 2000
circ = 10
xlist = np.linspace(-a/2,a/2,N)
dx = xlist[1] - xlist[0]
w1 = w0*np.sqrt(1+(d1/z0)**2)
w2 = w0*np.sqrt(1+(d2/z0)**2)

"""
we can somewhere add filters to choose the fundamental mode 
"""
rb = 1.5*w0
G_filter = np.abs(xlist) <= rb

RR1 = -(d1 + z0**2/d1)
RR2 = (d2 + z0**2/d2)
print(RR1,RR2)

#E = (xlist/w1)*np.exp(-(xlist/w1)**2) # E1_start
#E2 = (xlist/w2)*np.exp(-(xlist/w2)**2) # E1_start

E = ((xlist/w1)**2)*np.exp(-0.8*(xlist/w1)**2) # E1_start

mask_1 = np.exp(-1j*pi*xlist**2/(lbd*R1)) # + 
mask_2 = np.exp(-1j*pi*xlist**2/(lbd*R2)) # +

for n in range(circ):
    E_filt_1 = propagate(E,lbd,d1,dx,mask_1)
    E_g2 = propagate(E_filt_1*G_filter,lbd,d2,dx,1)*mask_2
    E_filt_2 = propagate(E_g2,lbd,d2,dx,mask_2)
    E = propagate(E_filt_2*G_filter,lbd,d1,dx,1)*mask_1

fig, ax = plt.subplots()

ax.plot(xlist,1.35*2.3*np.abs(E),label = "cavity iteration_1")
#ax.plot(xlist,2.55*np.abs(E_mid),label = "cavity iteration_2")

ax.plot(xlist,np.abs(eval_hermite(n_T,np.sqrt(2)*xlist/w1))*np.exp(-(xlist/w1)**2)*np.sqrt(w0/w1),label = f"hermite gaussian mode_1, n_T = {n_T}")
#ax.plot(xlist,np.abs(eval_hermite(n_T,np.sqrt(2)*xlist/w2))*np.exp(-(xlist/w2)**2)*np.sqrt(w0/w2),label = f"hermite gaussian mode_2, n_T = {n_T}")

ax.set_xlabel("x")
ax.set_ylabel("field")
ax.set_title(f"laser cavity, lbd = {lbd*1e9:.2f}nm")
ax.legend()

plt.show()