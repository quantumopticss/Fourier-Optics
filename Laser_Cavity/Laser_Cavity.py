"""
Finite size of reflector -> 

2D gaussian beam

differection loss & change field distribution 

"""
from ..units import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_hermite
from propagation_fresnel import propagate
## parameters
a = 2*cm
d = 1*m
R1 = -1.5*m
R2 = -1.5*m

# from spherical mirror cavity we know
z0 = np.sqrt(-(R1+d)*(R2+d)*(R1+R2*d)*d/((R1+R2*2*d)**2))
d1 = d*(R2+d)/(R1+R2+2*d)
d2 = (R1+d)*d/(R1+R2+2*d)

n_L = 20000 # longitudinal number
n_T = 2 # transverse number

freq = c0_const/(2*d) * (n_L + (n_T+0.5)*(np.arctan(d1/z0) + np.arctan(d2/z0))/pi) 
lbd = c0_const/freq
k = 2*pi*lbd

# U2(x,y) = 1j*np.exp(-1j*k*d)/(lbd*d) * iint U1(x',y')*np.exp(   -1j*pi/lbd*(  (x'**2 + y'**2)/R1 +  (x**2 + y**2)/R2  )  )
#                                                      *np.exp(-1j*pi/d*((x-x')**2) + (y-y')**2) dx'dy'

# operate
N = 2000
circ = 10
xlist = np.linspace(-a/2,a/2,N)
E = np.ones_like(xlist) # E_start

mask_1 = np.exp(-1j*pi*xlist**2/(lbd*R1))
mask_2 = np.exp(-1j*pi*xlist**2/(lbd*R2))

for n in range(circ):
    E_mid = propagate(E,d,lbd,mask_1)*mask_2
    E = propagate(E_mid,d,lbd,mask_2)*mask_1

fig, ax = plt.subplots()
ax.plot(xlist,E,label = "cavity iteration")
ax.plot(xlist,eval_hermite(),label = f"hermite gaussian mode, n_T = {n_T}")
ax.set_xlabel("x")
ax.set_title("laser cavity")
ax.legend()

plt.show()