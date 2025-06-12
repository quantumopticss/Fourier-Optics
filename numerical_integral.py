import numpy as np
from scipy.integrate import quad
### 1D integral
def integral_subcalculate(fun,x_start:float,x_end:float,args:tuple = ()) -> float:
    # calculate a small integrate over xspan

    Xlist4 = np.linspace(0,1,5,endpoint = True)*(x_end - x_start) + x_start
    i4 = fun(Xlist4,*args)
    Co_4 = np.array([7,32,12,32,7])

    I = (np.sum(Co_4*i4,axis = None))*(x_end - x_start)/90
    return I

def integral_operate(fun,xspan: np.ndarray,args:tuple = (),TOL:float = 1e-6) -> float:
    N = 8
    int_space = np.linspace(xspan[0],xspan[1],N,endpoint = True)
    result = 0
    j = 0
    n = 8
    while(j<(N-1)): ## integral in int_space
        n = np.max([N//2,n//2])
        erf = 1
        result_0 = integral_subcalculate(fun,int_space[j],int_space[j+1],args)
        while(erf >= TOL): ## self_adjust integral in span: { int_space[j],int_space[j+1] }
            sub_space = np.linspace(int_space[j],int_space[j+1],n,endpoint = True); 
            result_1 = 0
            k = 0
            while(k<(n-1)): ## integral in sub_space
                result_1 += integral_subcalculate(fun,sub_space[k],sub_space[k+1],args)
                k += 1
            n *= 2

            erf = np.abs(result_1 - result_0)/np.max([1,np.abs(result_1)])
            result_0 = result_1 

        result += result_0
        j += 1
    
    return result

def integral(fun,xspan: np.ndarray,args:tuple = (),TOL: float = 1e-6) -> float:
    """ numerical integral
        support complex function: np.sqrt(-1+0j) = 1j
        fun:  function to integrate,
        xpans: integral span, 
        TOL: value to control numerical error
    """
    if xspan[1] == xspan[0]:
        return 0
    
    if xspan[1] > xspan[0]:
        sgn = 1
        x0 = xspan[0]
        x1 = xspan[1]
    else:
        sgn = -1
        x0 = xspan[1]
        x1 = xspan[0]
    
    process = [0,0]
    if x1 == np.infty:
        process[1] = 1
    if x0 == -np.infty:
        process[0] = -1
    
    ## integral
    match process:
        case [0,1]:
            step = 1
            result = integral_operate(fun,[x0,x0+step],args,TOL)
            e = 1
            while(e>=TOL):
                x0 += step
                add = integral_operate(fun,[x0,x0+step],args,TOL)
                result += add
                e = np.abs( add/np.max([np.abs(result),1])  )

        case [-1,0]:
            step = 1
            result = integral_operate(fun,[x1-step,x1],args,TOL)
            e = 1
            while(e>=TOL):
                x1 -= step
                add = integral_operate(fun,[x1-step,x1],args,TOL)
                result += add
                e = np.abs( add/np.max([np.abs(result),1])  )

        case [-1,1]:
            # [0,1]
            x0 = 0
            step = 1
            result_f = integral_operate(fun,[x0,x0+step],args,TOL)
            e = 1
            x0 += step
            while(e>=TOL):
                x0 += step
                add = integral_operate(fun,[x0,x0+step],args,TOL)
                result_f += add
                e = np.abs( add/np.max([np.abs(result_f),1])  )

            #[-1,0]
            x1 = 0
            result_b = integral_operate(fun,[x1-step,x1],args,TOL)
            e = 1
            while(e>=TOL):
                x1 -= step
                add = integral_operate(fun,[x1-step,x1],args,TOL)
                result_b += add
                e = np.abs( add/np.max([np.abs(result_b),1])  )

            result = result_f + result_b

        case _:
            result = integral_operate(fun,[x0,x1],args,TOL)

    return (result*sgn)

### 2D integral
def integral2D_subcalculate(yxfun,funy_down,funy_up,x_start:float,x_end:float,args:tuple=()) -> float:
    xlist = np.linspace(x_start,x_end,5,endpoint = True)
    Co_4 = np.array([7,32,12,32,7])
    result = 0
    i = 0
    while(i<5):
        int_y = integral(yxfun,[funy_down(xlist[i]),funy_up(xlist[i])],(xlist[i],*args))
        result += int_y*Co_4[i]
        i += 1

    return (result*(x_end - x_start)/90)

def integral2D_operate(yxfun,funy_down,funy_up,xspan: np.ndarray,args:tuple=(),TOL:float = 1e-6) -> float:
    N = 8
    int_xspace = np.linspace(xspan[0],xspan[1],N,endpoint = True)
    result = 0
    j = 0
    n = 8
    while(j<(N-1)): ## integral in int_space
        n = np.max([N//2,n//2])
        erf = 1
        result_0 = integral2D_subcalculate(yxfun,funy_down,funy_up,int_xspace[j],int_xspace[j+1],args)
        while(erf >= TOL): ## self_adjust integral in span: { int_space[j],int_space[j+1] }
            sub_space = np.linspace(int_xspace[j],int_xspace[j+1],n,endpoint = True); 
            result_1 = 0
            k = 0
            while(k<(n-1)): ## integral in sub_space
                result_1 += integral2D_subcalculate(yxfun,funy_down,funy_up,sub_space[k],sub_space[k+1],args)
                k += 1
            n *= 2

            erf = np.abs(result_1 - result_0)/np.max([1,np.abs(result_1)])
            result_0 = result_1 

        result += result_0
        j += 1
    
    return result

def integral2D(yxfun,funy_down,funy_up,xspan:np.ndarray,args:tuple = (),TOL:float = 1e-6) -> float:
    """ 2D numerical integral
        support complex function: np.sqrt(-1+0j) = 1j
        yxfun: function to integrate, but you should use function of this form: func(y,x,*args)
        xpans: integral span of x-axis
        funy: funtion of y lines
        TOL: value to control numerical error
    """
    if xspan[1] == xspan[0]:
        return 0
    
    if xspan[1] > xspan[0]:
        sgn = 1
        x0 = xspan[0]
        x1 = xspan[1]
    else:
        sgn = -1
        x0 = xspan[1]
        x1 = xspan[0]
    
    process = [0,0]
    if x1 == np.infty:
        process[1] = 1
    if x0 == -np.infty:
        process[0] = -1
    
    ## integral -- integral2D_operate(yxfun,funy_down,funy_up,xspan,args,TOL)
    match process:
        case [0,1]:
            step = 1
            result = integral2D_operate(yxfun,funy_down,funy_up,[x0,x0+step],args,TOL)
            e = 1
            while(e>=TOL):
                x0 += step
                add = integral2D_operate(yxfun,funy_down,funy_up,[x0,x0+step],args,TOL)
                result += add
                e = np.abs( add/np.max([np.abs(result),1])  )

        case [-1,0]:
            step = 1
            result = integral2D_operate(yxfun,funy_down,funy_up,[x1-step,x1],args,TOL)
            e = 1
            while(e>=TOL):
                x1 -= step
                add = integral2D_operate(yxfun,funy_down,funy_up,[x1-step,x1],args,TOL)
                result += add
                e = np.abs( add/np.max([np.abs(result),1])  )

        case [-1,1]:
            # [0,1]
            x0 = 0
            step = 1
            result_f = integral2D_operate(yxfun,funy_down,funy_up,[x0,x0+step],args,TOL)
            e = 1
            while(e>=TOL):
                x0 += step
                add = integral2D_operate(yxfun,funy_down,funy_up,[x0,x0+step],args,TOL)
                result_f += add
                e = np.abs( add/np.max([np.abs(result_f),1])  )

            #[-1,0]
            x1 = 0
            result_b = integral2D_operate(yxfun,funy_down,funy_up,[x1-step,x1],args,TOL)
            e = 1
            while(e>=TOL):
                x1 -= step
                add = integral2D_operate(yxfun,funy_down,funy_up,[x1-step,x1],args,TOL)
                result_b += add
                e = np.abs( add/np.max([np.abs(result_b),1])  )

            result = result_f + result_b

        case _:
            result = integral2D_operate(yxfun,funy_down,funy_up,xspan,args,TOL)

    return (result*sgn)

### 1D discrete integral
def int_integral(ylist:np.ndarray,xlist:np.ndarray,order:int = 4,smooth:float = 0.02,TOL:float = 1e-6) -> float:
    """ discrete numerical integral, first do interpolation, and then numerical integrate
        it will return the integrate result and a spline function for you to check the vaildity

        ylist: discrete sampling date from sensor you want to integrate over xlist opi
        xlist: span over which you want to do integration about ylist
        order: value to control interpolate error 1 <= order <= 5.
        smooth: value to control the spline process: smooth >= 0
                    larger smooth value will get smoother cruves,  smaller value to be more like raw data
                   'smooth = None' to use the default value in scipy.UnivariateSpline (discourage)
        TOL: value to control integrate error

        the error is contributed both from numerical integration and spline progress, which will 
        be influenced by the accuracy and fluctuations of your sampling data
    """
    from scipy.interpolate import UnivariateSpline
    spline = UnivariateSpline(xlist,ylist,k = order,s = smooth) ## get interpolate function
    fun = lambda x: spline(x)

    result = integral(fun,[xlist[0],xlist[-1]],(),TOL)
    return result,fun

def fft(ylist:np.ndarray) -> np.ndarray:
    '''
    
    f_n = (1/N) * sum_{i = 0}^{N - 1} F_i * exp(-1i * 2*pi*n*i/N)
    
    F_n = sum_{n = 0}^{N - 1} f_j * exp(1i * 2*pi*n*j/N )
    
    traditional DFT algorithm for understanding how fft whorks
    we suggest to use np's fft function for numerical computation

    tlist: sampling time list
    ylist: data list, corresponding to tlist

    sampling frequenncy: f = 2*F_N, Nyquist frequency F_N is the frequency only under which can a phenomenon be analyzed by this fft
    frequency resolution: delta_f = 1/T, T is the whole time of tlist
    a[n], n = np.arange(0,,N), frequency list of fft
    '''
    N = len(ylist)
    nlist = np.arange(N) # 0,1,2,...,(N-1)
    an = np.empty_like(nlist,dtype = complex)
    
    philist = np.empty([N,N])
    for n in range(N):
        philist[:,n] = np.exp(1j*2*np.pi*n*nlist/N)
    
    an = ylist @ philist
    
    return an 

def fftfreq(length:int,delta_t:float) -> np.ndarray:
    '''
    lenth: lenth of fft list
    dt: sampling retardation between two neighbour data in tlist 
    '''
    val = 1/(length*delta_t)
    flist_p = np.arange(0,length//2)
    flist_n = np.arange(-length//2,0)

    flist = np.concatenate((flist_p,flist_n),axis = 0)*val
    return flist

def fftshift(list:np.ndarray) -> np.ndarray:
    """
    1D fftshift:
    
    shift fftlist so as to let the DC component at the center of the list
    """
    N = len(list)
    shift_list = np.concatenate((list[N//2:],list[0:N//2-1]),axis = 0)
    return shift_list

# testing functions  
def yxfun(y,x):
    f = y**2*np.exp(-x**2)
    return f

def fun(x):
    f = 2*np.cos(2*np.pi*x)
    return f

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    #print(np.sqrt(-1+0j)) --> 1j
    f = 100
    T = 4
    tlist = np.linspace(0,T,T*f)
    ylist = fun(tlist)

    yf = fftshift(fft(ylist))/len(ylist)
    tf = fftshift(fftfreq(len(ylist),1/f))
    print(np.sum(np.abs(yf)**2,axis = None))

    plt.figure(1)
    plt.plot(tf,np.abs(yf))
    plt.show()


    