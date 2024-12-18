import numpy as np
a = np.arange(0,10)

A = 5

a_b = (a<=A)
print(a_b)

indices = np.where(a_b)

print(indices)