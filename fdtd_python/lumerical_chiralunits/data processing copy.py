import numpy as np
import matplotlib.pyplot as plt

def dp_main():
    a = 120
    b = 30
    anglelist = np.linspace(0,np.pi/2,10,endpoint = True)
    result_tA = np.empty([2,np.size(anglelist)])
    result_tphi = np.empty_like(result_tA)     
    i = 0
    while(i<np.size(anglelist)): # load data
        fileID_x =  ("results/" + "EXresult_" + f"theta = {anglelist[i]:.2f},a = {(a):.2f},b = {(b):.2f}.txt")
        fileID_y =  ("results/" + "EYresult_" + f"theta = {anglelist[i]:.2f},a = {(a):.2f},b = {(b):.2f}.txt")
        #fileID_x =  ("results/" + "EXresult_" + M + f"_a = {(a):.2f},b = {(b):.2f}.txt")

        matrix_ey = np.loadtxt(fileID_y,dtype = complex)
        matrix_ex = np.loadtxt(fileID_x,dtype = complex)

        Ey = np.sum(matrix_ey,axis = None)/(np.size(matrix_ey))    
        Ex = np.sum(matrix_ex,axis = None)/(np.size(matrix_ex))        
        
        result_tA[:,i] = np.array([np.abs(Ex),np.abs(Ey)])
        result_tphi[:,i] = np.array([np.angle(Ex),np.angle(Ey)])
        i += 1

    result_tphi = np.mod(result_tphi,(2*np.pi))

    EE = result_tA*np.exp(1j*result_tphi)
    Exx = EE[0,:]*np.cos(anglelist) + EE[1,:]*np.sin(anglelist)
    Eyy = EE[0,:]*np.cos(anglelist) - EE[1,:]*np.sin(anglelist)

    ## figure 
    plt.figure(1)
    plt.plot(anglelist,result_tA[0,:],label = 'tx')
    plt.plot(anglelist,result_tA[1,:],label = 'ty')
    plt.plot(anglelist,result_tA[1,:] + result_tA[0,:],label = 'all')
    #plt.plot(anglelist,result_tphi[0,:],label = 'phix')
    #plt.plot(anglelist,result_tphi[1,:],label = 'phiy')
    plt.legend()

    plt.show()

if __name__ == "__main__":
    dp_main()