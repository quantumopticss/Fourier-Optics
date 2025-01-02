import numpy as np

a = np.array( [[1,1,1],[1,1,1],[1,1,1]] )
b = np.array([1,2,3])

c = b[np.newaxis,:]*a # differ in C index
print(np.roll(c,1,axis = 1)) # -> 3,1,2

A = (1,2)
a,b = A
print(a,b)

print(len(c.shape))