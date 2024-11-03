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
h1 = 40*nm # mesh accuracy 1
h2 = 40*nm # mesh accuracy 2

a_2 = 4*um # the radius with in which we will use h2 for mesh size

n1 = 1.4457 # core
n2 = 1.4378 # cladding
Delta = (n1**2 - n2**2)/(2*n1**2) # fiber Delta parameter
P = 2 # GRIN fiber P index 

L = 1 # azimuthal index
M = 1 # fiber radical order index
k0 = 2*pi/lbd # vaccuum wavevector

### define refractive index
if fiber == "GRIN_fiber":
    n_index = lambda rho: np.sqrt( n2**2 + (n1**2*( 1 - 2*Delta*((rho/a)**P)   ) - n2**2 )*(rho<=a) )
elif fiber == "step_index_fiber":
    n_index = lambda rho: n2 + (n1-n2)*(rho<=a)
else:
    raise ValueError("Wrong fiber")

### operate
rho2_list = np.arange(1e-6*h2,a_2,h2,dtype = np.longdouble)
rho1_list = np.arange(rho2_list[-1]+h1,A*a,h1,dtype = np.longdouble)
rho_list = np.concatenate((rho2_list,rho1_list),axis = 0)
N1 = np.size(rho1_list)
N2 = np.size(rho2_list)

rho_mat = np.zeros([N1+N2,N1+N2],dtype = np.longdouble)
rho_mat[np.arange(0,N1+N2),np.arange(0,N1+N2)] = rho_list

"""
f(x+2*h) = f + 2*h*f' + 2*h**2*f'' + 4*h**3*f'''/3 + 2*h**4*f[4]/3 + ... 
f(x+h) = f + h*f' + h**2*f''/2 + h**3*f'''/6 + h**4*f[4]/24 + ... 
f(x-h) = f - h*f' + h**2*f''/2 - h**3*f'''/6 + h**4*f[4]/24 + ... 
f(x-2*h) = f - 2*h*f' + 2*h**2*f'' - 4*h**3*f'''/3 + 2*h**4*f[4]/3 + ... 

d = [-1,0,1]/(2*h) 
d^2 = [1,-2,1]/(h**2) ~ [0,0,1,-2,1]/(h**2)

special process at h1 h2 boundary
f(x+a) = f + a*f' + a**2*f''/2 + a**3*f'''/6 + a**4*f[4]/24 + ... 
f(x-b) = f - b*f' + b**2*f''/2 - b**3*f'''/6 + b**4*f[4]/24 + ... 

f(x+2a) = f + 2a*f' + 2*a**2*f'' + 4*a**3*f'''/3 + 2*a**4*f[4]/3 + ... 
f(x-2b) = f - 2b*f' + 2*b**2*f'' - 4*b**3*f'''/3 + 2*b**4*f[4]/3 + ... 

a*f(x-b) + b*f(x+a) - (a+b)*f(x) = ab*(a+b)/2 * f'' + ab*(a**2 - b**2)/6*f''' + o(f'''')
a*f(x-2b) + b*f(x+2a) - (a+b)*f(x) = 2*ab*(a+b) f'' + 4*ab*(a**2 - b**2)/3''' + o(f'''')

"""
Mat_0 = np.zeros_like(rho_mat,dtype = np.longdouble)
Mat_0[0:N2-1,0:N2-1] = (-2*np.eye(N2-1) + np.eye(N2-1,k=1) + np.eye(N2-1,k=-1) )/(h2**2)
Mat_0[N2::,N2::] = (-2*np.eye(N1) + np.eye(N1,k=1) + np.eye(N1,k=-1) )/(h1**2)
# special case for h1,h2 boundary

# Mat_0[N2-1,N2-3:N2+2] = np.array([-h1,8*h1,-7*(h1+h2),8*h2,-h2],dtype = np.longdouble)/np.longdouble(2*h1*h2*(h1+h2))
Mat_0[N2-1,N2-2:N2+1] = 2*np.array([h1,-(h1+h2),h2],dtype = np.longdouble)/np.longdouble(h1*h2*(h1+h2))

Mat_0[N2-2,N2-1] = (1/h2)**2
Mat_0[N2,N2-1] = (1/h1)**2

Mat_2 = np.zeros_like(Mat_0,dtype = np.longdouble)
Mat_2[np.arange(0,N1+N2),np.arange(0,N1+N2)] = - (L**2 - 0.25)/(rho_list**2) + (n_index(rho_list)*k0)**2

Mat = Mat_0 + Mat_2
# print(issymmetric(Mat))

### solve
eigval, eigvec = eigh(Mat)

### plot
num = -M
beta = np.sqrt(eigval[num])
vec = eigvec[:,num]

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