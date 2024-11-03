import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from units import *
from scipy.linalg import eigh, issymmetric
from fiber_plot import fiber_plot

"""
(nabla**2 + n(rho)**2*k0**2 )E = 0

E = R(rho)*P(phi)*mp.exp(-1j*beta*z)

P(phi) = np.exp(-1j*L*phi)

d_rho[ rho * d_rho[R(rho)] ]/rho - L**2/(rho**2)*R(rho) + (n(rho)*k0)**2*R(rho) = beta**2*E(rho)

R(rho) = u(rho)/np.sqrt(rho)

-> u'' + (k**2 - (L**2 - 0.25)/rho**2 )*u = beta**2*u # suitable for L != 0 condition

"""
### users set

fiber = "GRIN_fiber"
# fiber = "GRIN_fiber"
num = 0 # solve index

a = 8*um # raidus of fiber core
A = 4 # amp of solve area
lbd = 852*nm # wavelength
h = 40*nm # mesh accuracy

n1 = 1.4457 # core
n2 = 1.4378 # cladding
Delta = (n1**2 - n2**2)/(2*n1**2) # fiber Delta parameter
P = 2 # GRIN fiber P index 

L = 1 # azimuthal index
M = 2 # fiber radical order index
k0 = 2*pi/lbd # vaccuum wavevector

### define refractive index
if fiber == "GRIN_fiber":
    n_index = lambda rho: np.sqrt( n2**2 + (n1**2*( 1 - 2*Delta*((rho/a)**P)   ) - n2**2 )*(rho<=a) )
elif fiber == "step_index_fiber":
    n_index = lambda rho: n2 + (n1-n2)*(rho<=a)
else:
    raise ValueError("Wrong fiber")

### operate
rho_list = np.arange(5e-7*h,a*A,h)
N = np.size(rho_list)

"""
f(x+2*h) = f + 2*h*f' + 2*h**2*f'' + 4*h**3*f'''/3 + 2*h**4*f[4]/3 + ... 
f(x+h) = f + h*f' + h**2*f''/2 + h**3*f'''/6 + h**4*f[4]/24 + ... 
f(x-h) = f - h*f' + h**2*f''/2 - h**3*f'''/6 + h**4*f[4]/24 + ... 
f(x-2*h) = f - 2*h*f' + 2*h**2*f'' - 4*h**3*f'''/3 + 2*h**4*f[4]/3 + ... 

d = [-1,0,1]/(2*h) 
d^2 = [1,-2,1]/(h**2) ~ [0,0,1,-2,1]/(h**2)

"""
Mat_0 = (-2*np.eye(N,k=0) + np.eye(N,k=1) + np.eye(N,k=-1))/(h**2) #** [1,-2,1]/(h**2)
# Mat_0 = ( -30*np.eye(N,k=0) + 16*np.eye(N,k=1) + 16*np.eye(N,k=-1) - np.eye(N,k=2) - np.eye(N,k=-2) )/(12*h**2) #** [-1,16,-30,16,-1]/(12*h**2)

Mat_2 = np.zeros_like(Mat_0)
Mat_2[np.arange(0,N),np.arange(0,N)] = - (L**2 - 0.25)/(rho_list**2) + (n_index(rho_list)*k0)**2

Mat = Mat_0 + Mat_2
# print(issymmetric(Mat))

### solve
eigval, eigvec = eigh(Mat)

### plot
num = -M
beta = np.sqrt(eigval[num])
vec = eigvec[:,num]/np.sqrt(rho_list*h*1e9)

print(f"effective mode index = {beta/k0}")
print(fiber_plot(opc = "calculate",L = L,M = M,lbd = lbd, a = a,n1 = n1,n2 = n2))

fig, ax = plt.subplots()

line0 = ax.plot(rho_list*1e6,vec,label = "R(rho)",color = "red")
ax1 = ax.twinx()
line1 =  ax1.plot(rho_list*1e6,n_index(rho_list),label = "n(rho)")
ax.set_xlabel("x/[um]")
ax.set_ylabel("R(rho)")

lines = line0 + line1  # Concatenate the lines from both axes
labels = [line.get_label() for line in lines]  # Extract labels from the lines
ax.legend(lines, labels)  # Display the combined legend on one of the axes

ax.set_title(f"effective mode index = {beta/k0}")

plt.show()