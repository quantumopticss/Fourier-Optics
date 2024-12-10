import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft

d_nu = 1
N = 100
nulist = np.arange(-N//2,N//2)*d_nu
tlist = np.arange(0,N)*(1/(N*d_nu))

f0 = 20

f_nu = np.exp(-(nulist/f0)**2)