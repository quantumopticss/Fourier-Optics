import sys, os
path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(path)))

import numpy as np
import matplotlib.pyplot as plt
from units import *

## TE mode
n1 = 1.5
n2 = 1.2
d1 = 1*um
d2 = 1*um
ng = 1
lbd = 780*nm

k0 = 2*pi/lbd
theta = np.linspace(0,85,10)*deg

theta1 = np.arcsin(ng*np.sin(theta)/n1)
theta2 = np.arcsin(ng*np.sin(theta)/n2)

phi_1 = k0*d1*n1*np.cos(theta1)
phi_2 = k0*d1*n1*np.cos(theta2)

N1 = n1*np.cos(theta1)
N2 = n2*np.cos(theta2)

alpha = phi_1 + phi_2
beta = phi_1 - phi_2

Na = (N1+N2)/(2*np.sqrt(N1*N2))
Nb = (N1-N2)/(2*np.sqrt(N1*N2))

Lp = Na**2*np.cos(alpha) - Nb**2*np.cos(beta) + np.sqrt( (Na**2*np.cos(alpha) - Nb**2*np.cos(beta))**2 - 1 + 0j)
Ln = Na**2*np.cos(alpha) - Nb**2*np.cos(beta) - np.sqrt( (Na**2*np.cos(alpha) - Nb**2*np.cos(beta))**2 - 1 + 0j)

r12 = np.exp(1j*(alpha+beta)/2)*(2*1j*Na*Nb*np.sin((alpha+beta)/2))/( np.sqrt( (Na**2*np.cos(alpha) - Nb**2*np.cos(beta))**2 - 1 + 0j) + 1j*(Na**2*np.sin(alpha) - Nb**2*np.sin(beta)) )

plt.figure(1)
plt.plot(theta,np.abs(r12))
plt.xlabel("theta")
plt.show()
