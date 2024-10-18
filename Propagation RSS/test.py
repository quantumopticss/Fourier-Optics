import numpy as np

a = np.array([[1,2],[3,4]])
b = np.array([1,2])

b1 = b[np.newaxis,:]
b2 = b[:,np.newaxis]

print(b1*a)
print(b2*a)