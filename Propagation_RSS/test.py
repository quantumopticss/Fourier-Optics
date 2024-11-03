import numpy as np

N = 128
alpha = 0.2

nlist = np.arange(N) - N//2

f = lambda x: (np.exp(1j*2*np.pi*alpha*x) + np.exp(-1j*np.pi*alpha*x) ) 

f_n = np.fft.fftshift(np.fft.fft(f(nlist)))/N

ff = ( np.arange(0,N) - N//2 ) /(N)

import matplotlib.pyplot as plt

plt.figure(1)

plt.plot(ff,np.abs(f_n),label = "raw - real")
plt.title(f"real - alpha = {alpha}")
plt.legend()

plt.show()