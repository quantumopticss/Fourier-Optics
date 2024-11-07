import numpy as np

a = np.array([0,1,2])
b = np.array([1,2,3])

c = np.tensordot(a,b,axes = 0)
print(c)

# import numpy as np 
# import matplotlib.pyplot as plt

# a1 = np.array([0.25,3,5,4,2,2,3,3,1,2,1])
# p1 = np.array([85,87,97,90.4,71,82.6,99.3,94,91,96,95])

# a2 = np.array([3,0.25,5,2,2,1,2,2,3,3,2,2,2,1,3])
# p2 = np.array([83,86,94,87,90,87,90,90,100,92,91,100,90,88.8,96])

# a3 = np.array([0.25,2,2,1,2,2,3,1,4,4,1,2,2])
# p3 = np.array([90,93,95,93,90,91,90,93,99,97,86,92,95])

# a4 = np.array([0.25,1,3,1,1,4,1,2,2,4,3,4])
# p4 = np.array([90,95,86,95,96,98,92,96,86,99,100,98])

# A1 = a1
# P1 = p1

# A2 = np.concatenate((a1,a2),axis = 0)
# P2 = np.concatenate((p1,p2))

# A3 = np.concatenate((a1,a2,a3))
# P3 = np.concatenate((p1,p2,p3))

# A4 = np.concatenate((a1,a2,a3,a4))
# P4 = np.concatenate((p1,p2,p3,p4))

# g1 = np.sum(A1*P1/20,axis = None)/np.sum(A1)
# g2 = np.sum(A2*P2/20,axis = None)/np.sum(A2)
# g3 = np.sum(A3*P3/20,axis = None)/np.sum(A3)
# g4 = np.sum(A4*P4/20,axis = None)/np.sum(A4)

# gpa_list = np.array([g1,g2,g3,g4])

# plt.figure(1)
# plt.plot([1,2,3,4],gpa_list)
# plt.xlabel('terms')
# plt.title('GPA')
# plt.show()