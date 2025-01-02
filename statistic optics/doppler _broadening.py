import sys, os
path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(path)))

import numpy as np
from units import *
import matplotlib.pyplot as plt
from scipy.special import erfinv

## maxwell velocity disperbution
## f(v) = np.sqrt(m/(2*pi*k*T)) * np.exp( -m*V**2/(2*k*T) )
## u = np.sqrt(2*k*T/m)
## f(v) = 1/( sqrt(pi) * u) * np.exp(-(v/u)**2)
## -> P(v) = \int_{0}^{v} f(v) dv = 1/sqrt(pi) * int_{0}^{v/u} exp(-x**2) dx = 0.5 * erf(v/u) = rand() - 0.5
## -> v = u * erfinv( 2*rand() - 1 )

## erf = int_{0}^{x} 2/sqrt(pi) * exp(-x^2) dx

# nu' = nu0 * (1 + v/c) ; dnu' = nu0/c * dv ; v = (nu'/nu0 - 1)*c
# dp = g*dnu' = f*dv = f*c/nu0 * dnu' -> g = c/nu0 * f
# g = 1/(u*sqrt(pi)) * np.exp( (c/u)**2*(nu'/nu - 1)**2 ) * c/nu0

T = 90000000000000 # K
mu_m = 12*1e-3*kg
num = 30000 # num

E0 = 1*V/m
nu0 = 370*Hz # central freq
f_samp = 1000*Hz 

tau = 2*ms # lifetime
N = int(100*tau*f_samp)
tlist,dt = np.linspace(-tau*10,tau*10,N,retstep=True)

Elist_c = np.zeros_like(tlist,dtype = complex)
Elist_d = np.copy(Elist_c)

## operate
u = np.sqrt(2*8.31*T/mu_m)
beta = u/c0_const

for _ in range(num):
    p = np.random.rand()
    v = u * erfinv(2*p - 1.)
    
    nu = nu0 * (1 + v/c0_const)
    Elist_c = Elist_c + E0 * np.exp(-np.abs(tlist)/tau) * np.exp(1j*2*pi*nu*tlist)
    Elist_d = Elist_d + E0 * np.exp(1j*2*pi*nu*tlist)
    
Elist_0 = num*E0*np.exp(-np.abs(tlist)/tau)*np.exp(1j*2*pi*nu0*tlist)

f_E0 = np.abs(np.fft.fftshift(np.fft.fft(Elist_0))*dt)**2
f_Ec = np.abs(np.fft.fftshift(np.fft.fft(Elist_c))*dt)**2
f_Ed = np.abs(np.fft.fftshift(np.fft.fft(Elist_d))*dt)**2
f_nu = np.arange(-N//2,N//2)/(20*tau)
d_nu = 1/(20*tau)

g = c0_const/(nu0*u*np.sqrt(pi)) * np.exp(-(c0_const*(f_nu/nu0 -1)/u)**2)

# ax.plot(f_nu,np.abs(f_E0),label = "nature")
# ax.plot(f_nu,np.abs(f_Ec),label = "dopper + nature")
fig, ax = plt.subplots()
ax.plot(f_nu,f_Ed,label = "dopper") # -> E0**2 * num**2 
ax.plot(f_nu,(g*num*E0)**2,label = "dopper - cal") # g * d_nu = d_p
ax.set_xlabel("freq/Hz")
ax.set_ylabel("amp")
ax.set_title(f"doppler broadening @ T = {T}K, vp = {1e4*beta:.3f}*1e-4*c")
ax.legend()

fig1, ax1 = plt.subplots()
ax1.plot(f_nu,np.abs(f_Ec),label = "dopper + nature") # -> E0**2 * num**2 
ax1.set_xlabel("freq/Hz")
ax1.set_ylabel("amp")
ax1.set_title(f"nature + doppler broadening @ T = {T}K, vp = {1e4*beta:.3f}*1e-4*c")
ax1.legend()

plt.show()