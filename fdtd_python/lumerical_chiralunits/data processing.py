import numpy as np
import matplotlib.pyplot as plt

def dp_main():
    M = "TE"
    alist = np.linspace(50,150,6,endpoint = True)
    blist = np.linspace(50,150,6,endpoint = True)
    
    result_tA = np.empty([np.size(alist),np.size(blist)])
    result_tphi = np.empty_like(result_tA)     
    ia = 0
    while(ia<np.size(alist)):
        ib = 0
        while(ib<np.size(blist)):
            # load data
            a = alist[ia]
            b = blist[ib]
            fileID_y =  ("results/" + "EYresult_" + M + f"_a = {(a):.2f},b = {(b):.2f}.txt")
            #fileID_x =  ("results/" + "EXresult_" + M + f"_a = {(a):.2f},b = {(b):.2f}.txt")

            matrix_ey = np.loadtxt(fileID_y,dtype = complex)
            #matrix_ex = np.loadtxt(fileID_x,dtype = complex)
            ty = np.sum(matrix_ey,axis = None)/(np.size(matrix_ey))          
            result_tA[ia,ib] = np.abs(ty)
            result_tphi[ia,ib] = np.angle(ty)

            ib +=1
        ia += 1

    ## figure 
    result_tA = result_tA.T
    result_tphi = result_tphi.T
    plt.subplot(1,2,1)
    plt.imshow(result_tA,extent = [alist[0],alist[-1],blist[0],blist[-1]],origin='lower', cmap='viridis')
    plt.xlabel('a')
    plt.ylabel('b')
    plt.title(M + "Mode")
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.subplot(1,2,2)
    plt.imshow(result_tphi,extent = [alist[0],alist[-1],blist[0],blist[-1]],origin='lower', cmap='twilight')
    plt.xlabel('a')
    plt.ylabel('b')
    plt.title(M + " Mode")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.show()

if __name__ == "__main__":
    dp_main()