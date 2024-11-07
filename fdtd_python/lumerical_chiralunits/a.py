import numpy as np

unit = 1e-6
W = 1.2*unit
Num = 4
T = 0.5

M_matrix = np.zeros([Num,Num],dtype = int)
M_matrix[2:7+1:1,2:5+1:1] = 1

ylist = ( np.arange(0,Num,1) - (Num-1)/2 )* (W/Num)
xlist = -ylist

location_X, location_Y = np.meshgrid(ylist,xlist)

period = 2*unit
h = 1*unit
epoch = 1
score_old = 0

print(M_matrix)
index = np.random.randint(0,2,size = [2])
print(index)
print(M_matrix[index[0],index[1]])