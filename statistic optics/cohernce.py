import numpy as np
import matplotlib.pyplot as plt

c0_const = 3*1e8
nu_0 = 500*1e12
delta_nu = 2.5*1e12

xlist = np.linspace(-1e-6,1e-6,500)*80

f1 = lambda x : np.exp(-(np.pi*delta_nu*2*x/c0_const)**2)*np.cos(2*np.pi*nu_0*x/c0_const)
f2 = lambda x : np.exp(-np.abs(4*np.pi*delta_nu*x/c0_const))*np.cos(2*np.pi*nu_0*x/c0_const)
f3 = lambda x : (np.sin((4*np.pi*delta_nu*x/c0_const))/(4*np.pi*delta_nu*x/c0_const))*np.cos(2*np.pi*nu_0*x/c0_const)
f4 = lambda x : (1+2*np.cos(4*np.pi*x*delta_nu/c0_const)/3)*np.cos(2*np.pi*nu_0*x/c0_const)

name = ["gaussian","lorentz","rect","comb"]

for i in range(4):
    plt.subplot(2,2,i+1)
    plt.plot(xlist*1e6,(1+eval(f"f{i+1}(xlist)"))/2)
    plt.xlabel("x/um")
    plt.ylabel("I(x)")
    plt.title(name[i])
    
plt.show()


