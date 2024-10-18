"""
here we will introcude some basics about torch
"""
import torch as th
import numpy as np

print(th.__version__)

device = "cuda" if th.cuda.is_available() else "cpu"
print(device)

a = th.tensor([[1,2,3],[4,5,6]],dtype = th.uint8,device = device)
print(a[0,1]) # -> 2

b = th.zeros_like(a) # all 1
c = th.rand((2,3)) # -> [[A1,A2,A3],[A4,A5,A6]]

d = c.numpy() # th.tensor can't be operated by function defined in numpy
print(np.fft.fft(d)[0])

# th.fft.fft() series
