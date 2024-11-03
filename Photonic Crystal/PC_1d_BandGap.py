import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig

## computation of 1D photonic crystal
## dispersion relation and E-field distribution

## able to solve on axis mode and off-axis but TM mode

def pc_brillioun_zone(x):
    if x <= 1:
        k = x
        kx = 0
        
        
    if x>1 and x<=2:
        kx = (x - 1)
        k = 1
        
    if x > 2:
        k = (3 - x)
        kx = k
        
    return k/2,kx/2
    
def pc_main():
    ## users set
    n1 = 1.5
    n2 = 3.5
    d1 = 1
    d2 = 1
    
    N_samp = 2**(6)

    ## operation
    num = 5
    
    nfun = lambda x: n1*(x <= d1)*(x>=0) + n2*(x > d1)*(x<=(d2+d1))
    
    xlist = np.linspace(0,d1+d2,2*N_samp,endpoint = False)
    xlist_plot = np.linspace(0,d1+d2,N_samp,endpoint = False)
    eta_list = 1/(nfun(xlist)**2)

    f_eta = np.fft.fftshift(np.fft.fft(eta_list))/(2*N_samp)
    
    F_eta = f_eta[N_samp]*np.eye(N_samp,k=0,dtype = np.complex128)
    for i in range(1,N_samp):
        F_eta = F_eta + f_eta[N_samp + i]*np.eye(N_samp,k=-i,dtype = np.complex128) + f_eta[N_samp - i]*np.eye(N_samp,k=i,dtype = np.complex128)

    ## different F matrix for different bloch wavevector k
    xlist = np.arange(0,1,0.02) # k
    ylist = np.empty([num,len(xlist)])

    for order in range(len(xlist)):
        x = xlist[order]
        k, kx_proportion = pc_brillioun_zone(x)
        
        f_n = np.arange(0,N_samp) - N_samp//2 + k 
        F_aa = np.tensordot(f_n,f_n,axes = 0)*F_eta # 
        
        F_ab = np.tile((k + np.arange(-N_samp//2,N_samp//2)),(N_samp,1)).T
        F_ab = -F_ab*k*kx_proportion*F_eta
              
        F_ba = np.tile((k + np.arange(-N_samp//2,N_samp//2)),(N_samp,1))
        F_ba = -kx_proportion*k*F_eta*F_ba
        
        F_bb = (kx_proportion*k)**2*F_eta
        
        FF = np.empty([2*N_samp,2*N_samp],dtype = np.complex128)
        FF[0:N_samp,0:N_samp] = F_aa
        FF[0:N_samp,N_samp::] = F_ab
        FF[N_samp::,0:N_samp] = F_ba
        FF[N_samp::,N_samp::] = F_bb       
        
        A = np.sum(np.abs(FF - np.conj(FF.T)),axis = None)

        eig_val, eig_vec = eig(FF)  
        ylist[:,order] = np.abs(np.sqrt(eig_val[0:num]))
        

    ## plot
    # plt.figure(1)
    # plt.plot(xlist,nfun(xlist),label = "refractive index")
    # plt.plot(xlist,(np.fft.ifft(np.fft.fftshift(f_eta))*N_samp*2)**(-0.5),label = "fft refractive index")
    # plt.plot(xlist_plot,np.abs(P_a),label = "E-field magnitude_a")
    # plt.plot(xlist_plot,np.abs(P_b),label = "E-field magnitude_b")
    # plt.xlabel('x')
    # plt.legend()
    # plt.title("refractive index & field distribution")
    
    plt.figure(2)
    for i in range(num):
        plt.scatter(xlist,ylist[i,:],label = f"numerical dispersion relation - {i+1}")
    plt.xlabel('Bloch Wavevector in g')
    plt.title('numerical omega in omega_b') # omega_b := 2*pi/(d1+d2)
    plt.legend()
    
    # plt.figure(3)
    # a = 2*np.pi/(d1+d2)
    # wlist_1 = np.linspace(0.001,0.156,25)
    # wlist_2 = np.linspace(0.24,0.349,25)
    # wlist_3 = np.linspace(0.452,0.595,25)
    # K_bloch = lambda w: 1/(2*np.pi)*np.arccos( (n1+n2)**2/(4*n1*n2) * ( np.cos(w*a*(n1*d1+n2*d2)) - (n1-n2)**2/(n1+n2)**2 * np.cos(w*a*(n1*d1 - n2*d2)) )      )    
    # plt.plot(K_bloch(wlist_1),wlist_1,label = "ana dispersion relation - 1")
    # plt.plot(K_bloch(wlist_2),wlist_2,label = "ana dispersion relation - 2")
    # plt.plot(K_bloch(wlist_3),wlist_3,label = "ana dispersion relation - 3")
    # plt.xlabel('Bloch Wavevector in g')
    # plt.title('analytical omega in omega_b')
    # plt.legend()
    
    plt.show()
if __name__ == "__main__":
    pc_main()