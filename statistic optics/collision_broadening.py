import sys, os
path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(path)))

import numpy as np
from units import *
import matplotlib.pyplot as plt

nu0 = 1000*Hz

tau = 30*ms # lifetime
# E = E0*np.exp(-t/tau)*np.exp(1j*2*pi*nu*t) with natural linewidth delta_nu = 1/(2*pi*tau)

f = 3000*Hz # sampling frequency
f_col = 100*Hz # collision frequency
T = 60*tau # total samping time

N = int(2*f*T) # sampling points
tlist, delta_t = np.linspace(-T,T,N,retstep=True,endpoint=False)

num = 10000 # times
E0 = 100*V/m
col_count = np.zeros([num])

eff:int = 10 # effect cal
E_list = np.zeros_like(tlist,dtype = complex)
for j in range(num):
    phi_list = 2*pi*nu0*tlist + np.random.rand()*2*pi # generate a perfect phase
    # collision broadening
    p = np.random.rand(len(phi_list))
    p_add = (p <= delta_t*f_col)
    
    add_indices = np.where(p_add)[0]
    for i in add_indices:
        phi_list[i::] = phi_list[i::] + np.random.rand()*2*pi 
    col_count[j] = np.sum(p_add) # collision count
    
    E_list += E0*np.exp(-np.abs(tlist)/tau)*np.exp(1j*phi_list)
    
print(f"average collision = {np.mean(col_count):.2f}")    
print(f"f_col*tau = {f_col*tau:.2f}")

f_E = np.abs(np.fft.fftshift(np.fft.fft(E_list))*delta_t)
f_nu = np.arange(-N//2,N//2)/(2*T)

## width
half = np.max(f_E)/2
f_area = f_nu[f_E>=half]

f_left = np.min(f_area)
f_right = np.max(f_area)

plt.figure(1)
plt.plot(tlist[20000:40000]*1000,np.abs(E_list[20000:40000]),label = f"real E @ t")
plt.legend()
plt.xlabel("time/ms")
plt.ylabel("E amp")
plt.title(f"collision broadening, with tau = {tau*1000:.3f}ms, f_col = {f_col:.3f}Hz")
plt.axvline(x=f_left, color='red', linestyle='--', label=f'FWHM Left: {f_left:.2f}')
plt.axvline(x=f_right, color='red', linestyle='--', label=f'FWHM Right: {f_right:.2f}')

from scipy.optimize import curve_fit
def f_func(x,gamma,A,nu0):
    f = A * gamma/(2*pi) * 1/( (x-nu0)**2 + (gamma/2)**2 )
    return f

P0 = np.array([f_col+1/(tau),half*2,nu0])
p_opt, _ = curve_fit(f_func,f_nu,f_E,p0=P0)

plt.figure(2)
plt.plot(f_nu,np.abs(f_E),label = f"w = {1/(pi*tau):.3f}Hz, FWHM = {(f_right - f_left):.3f}Hz")
plt.plot(f_nu,f_func(f_nu,*p_opt),label = f"lorentz fit, width = {p_opt[0]}")
plt.legend()
plt.xlabel("freq")
plt.ylabel("relative amp")
plt.title(f"collision broadening, with tau = {tau*1000:.3f}ms, f_col = {f_col:.3f}Hz")

plt.show()