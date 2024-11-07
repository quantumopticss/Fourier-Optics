from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt

a = np.linspace(0,15,100)
func = lambda x: x**3/(np.exp(x) - 1)

b = np.zeros_like(a)
for i in np.arange(1,len(b)):
    res, _ = quad(func,0.,a[i])
    b[i] = res
    
plt.figure(1)
plt.plot(a,b,label = "integrate")
plt.plot(a,np.pi**4/15*np.ones_like(a),label = "constant")
plt.xlabel("x")
plt.legend()
plt.show()
