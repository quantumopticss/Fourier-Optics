import torch as th
from torch import nn
import matplotlib.pyplot as plt
from units import *

class Deep_GS(nn.Module):
    """
    g(x,y) = 1/(lbd*f) * F(x/(lbd*f),y/(lbd*f))
    """
    def __init__(self):
        self.dx = 10*um
        self.shape = [1024,1024]
        self.lbd = 780*nm
        self.f = 300*mm
        
        N, M = self.shape
        self.freq_i = th.arange(-N//2,N//2)/(N*self.dx)
        self.freq_j = th.arange(-M//2,M//2)/(M*self.dx)
        
        self.xx, self.yy = self.