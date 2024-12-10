from lab_optimizer import global_optimize
from lab_optimizer.units import *

import numpy as np

## here we use lab_optimization to "optimize" GRIN fiber

from fiber_solve import fiber

def main():
    a = 8*um # raidus of fiber core
    A = 4 # amp of solve area
    lbd = 420*nm # wavelength
    N = 1024 # mesh accuracy

    n1 = 1.5 # core
    n2 = 1.4 # cladding
    delta_n = n1 - n2 # fiber Delta parameter
    # P = 2 # GRIN fiber P index here we want to using opt to cal this

    L = 9 # azimuthal index
    M = 10 # fiber radical order index
    
    ### define refractive index
    def fiber_mode_dispersion(x):
        n_index = lambda rho: n2 + delta_n*(rho<=a) * ( x[0] * (1-(rho/a)**2) + x[1] * (1-(rho/a)**4) + (1 - x[0] - x[1]) * (1-(rho/a)**6) )

        f = fiber(n_index,lbd = lbd,r_max = a*A,N_mesh = N)
        eigs = np.array([])
        for i in range(1,L+1):
            f.mode_cal(L = i, mode_num = M)
            eigs = np.hstack((eigs,f.eig_val))
            
        neff = np.sqrt(eigs)/f.k0
        cost = np.std(neff)/np.min(neff)
        cost_dict = dict(cost = cost,uncer = 0,bad = None)
        return cost_dict
    
    opt = global_optimize(fiber_mode_dispersion,paras_init = np.array([2., 1.0]), bounds = ((0.5,10.),(0.5,10.0)),delay = 0.01,max_run = 50,method = "simplex")
    opt.optimization()
    opt.visualization()
    
if __name__ == "__main__":
    main()