import numpy as np

## explicit odesolvers
def ode23(fun,t_start: float,initial: np.ndarray,t_end: float,args:tuple=(),step_max: float = 1e-2,TOL: float = 1e-5) -> tuple[np.ndarray,np.ndarray]:
    """ 
    explicit ode solver using RK23 method

    return tlist, xlist 

    solutions of different time are stacked in axis = 0

    t_N = tlist[N,0] ; x_N = xlist[N,...]  

    t_N = tlist[N,0] ; x_N = xlist[N,...]  
    
    ***************************************
    fun: fun(t,x,*args), return dx/dt function of the ode function to solve
    t_start: start of the 'time'
    initial: initial condition
    t_end: end of the time
    step_max: max time step for ode solving
    TOL: value to control the error
    """
    ## begin and validity check
    if t_start >= t_end:
        return np.array([t_start]), np.array([initial])

    t = t_start
    h = step_max*1e-4
    x = initial
    tlist = np.zeros([1,1])
    tlist[0,0] = t
    xlist = np.array([initial])
    auxi = np.ones_like(x)

    ## operate
    while(t < t_end):
        ## ode23 body
        dxdt_1 = fun(t,x,*args)
        dxdt_2 = fun(t+0.5*h,x+0.5*h*dxdt_1,*args)
        dxdt_3 = fun(t+h,x+h*(-dxdt_1+2*dxdt_2),*args)
        
        delta_2 = 0.5*h*(dxdt_1+dxdt_3)
        delta_3 = h*(dxdt_1 + 4*dxdt_2 + dxdt_3)/6
        ## justify
        e = delta_3 - delta_2
        f = np.max([np.abs(x),auxi],axis = 0)
        rato = np.abs(e/f)

        ## results
        Z = np.max(rato,axis = None)/TOL
        if Z < 1:
            x = x + delta_3
            t += h
            tlist = np.vstack((tlist,t))
            xlist = np.concatenate((xlist,np.array([x])),axis = 0)
            h = np.min([(h/(Z+TOL**2))**(1/3),step_max])
        else:
            h *= 0.7*((1/(Z+TOL**2))**(1/3))
            continue
        
    ## return
    return tlist,xlist

def ode45(fun,t_start: float,initial: np.ndarray,t_end: float,args:tuple=(),step_max: float = 1e-2,TOL: float = 1e-5) -> tuple[np.ndarray,np.ndarray]:
    """ 
    explicit ode solver using RK45 method

    return tlist, xlist 

    solutions of different time are stacked in axis = 0

    t_N = tlist[N,0] ; x_N = xlist[N,...]  

    t_N = tlist[N,0] ; x_N = xlist[N,...]  
    
    ***************************************
    fun: fun(t,x,*args), return dx/dt function of the ode function to solve
    t_start: start of the 'time'
    initial: initial condition
    t_end: end of the time
    step_max: max time step for ode solving
    TOL: value to control the error
    """
    ## begin and validity check
    if t_start >= t_end:
        return np.array([t_start]), np.array([initial])

    t = t_start
    h = step_max*1e-4
    x = initial
    tlist = np.zeros([1,1])
    tlist[0,0] = t
    xlist = np.array([initial])
    auxi = np.ones_like(x)

    ## operate
    while(t < t_end):
        ## ode45 body
        dxdt_1 = fun(t,x,*args)
        dxdt_2 = fun(t+0.25*h,x+0.25*h*dxdt_1,*args)
        dxdt_3 = fun(t+0.375*h,x+h*(3*dxdt_1+9*dxdt_2)/32,*args)
        dxdt_4 = fun(t+ 12*h/13,x+h*(1932*dxdt_1 - 7200*dxdt_2 +7296*dxdt_3)/2197,*args)
        dxdt_5 = fun(t+h,x+h*(439*dxdt_1/216 - 8*dxdt_2 + 3680*dxdt_3/513 - 845*dxdt_4/4104),*args)
        dxdt_6 = fun(t+0.5*h,x+h*(-8*dxdt_1/27 + 2*dxdt_2 - 3544*dxdt_3/2565 + 1895*dxdt_4/4104 - 0.275*dxdt_5),*args)
        
        delta_4 = h*(25*dxdt_1/216 + 1408*dxdt_3/2565 + 2197*dxdt_4/4104 - 0.2*dxdt_5)
        delta_5 = h*(16*dxdt_1/135 + 6656*dxdt_3/12825 + 28561*dxdt_4/56430 - 0.18*dxdt_5 + 2*dxdt_6/55)
        ## justify
        e = delta_5 - delta_4
        f = np.max([np.abs(x),auxi],axis = 0)
        rato = np.abs(e/f)

        ## results
        Z = np.max(rato,axis = None)/TOL
        if Z < 1:
            x = x + delta_5
            t += h
            tlist = np.vstack((tlist,t))
            xlist = np.concatenate((xlist,np.array([x])),axis = 0)
            h = np.min([h*(1/(Z+TOL**2))**(1/3),step_max])
        else:
            h *= 0.7*((1/(Z+TOL**2))**(1/3))
            continue
        
    ## return
    return tlist,xlist

#def ode89(fun,t_start,initial,t_end,args=(),step_max = 1e-2,TOL = 1e-5):

def odeint(fun,t_start: float,initial: np.ndarray,t_end: float,args:tuple=(),t_step:float = 1e-1 ,step_max: float = 1e-2,TOL: float = 1e-5) -> tuple[np.ndarray,np.ndarray]:
    """ 
    explicit ode solver using RK45 method and will output the result in equal time step

    return tlist, xlist 

    solutions of different time are stacked in axis = 0

    t_N = tlist[N,0] ; x_N = xlist[N,...]  

    ***************************************
    fun: fun(t,x,*args), return dx/dt function of the ode function to solve
    t_start: start of the 'time'
    initial: initial condition
    t_end: end of the time
    t_step: time step of the output timelist
    step_max: max time step for ode solving
    TOL: value to control the error
    """
    if t_start >= t_end:
        return np.array([t_start]), np.array([initial])

    tlist0,xlist0 = ode45(fun,t_start,initial,t_end,args,step_max,TOL)

    tlist = np.arange(t_start,t_end+t_step,t_step)
    tlist.reshape(-1,1) # to [n,1] array
    xlist = np.array([initial])
    i = 1
    while(i < np.size(tlist)):
        n = np.sum((tlist0 <= tlist[i]),axis = None) - 1 # location of the nearest before time and step is lower than the TOLed step
        
        t = tlist0[n]
        h = tlist[i] - tlist0[n]
        x = xlist0[n]

        dxdt_1 = fun(t,x,*args)
        dxdt_2 = fun(t+0.25*h,x+0.25*h*dxdt_1,*args)
        dxdt_3 = fun(t+0.375*h,x+h*(3*dxdt_1+9*dxdt_2)/32,*args)
        dxdt_4 = fun(t+ 12*h/13,x+h*(1932*dxdt_1 - 7200*dxdt_2 +7296*dxdt_3)/2197,*args)
        dxdt_5 = fun(t+h,x+h*(439*dxdt_1/216 - 8*dxdt_2 + 3680*dxdt_3/513 - 845*dxdt_4/4104),*args)
        dxdt_6 = fun(t+0.5*h,x+h*(-8*dxdt_1/27 + 2*dxdt_2 - 3544*dxdt_3/2565 + 1895*dxdt_4/4104 - 0.275*dxdt_5),*args)

        x = x + h*(16*dxdt_1/135 + 6656*dxdt_3/12825 + 28561*dxdt_4/56430 - 0.18*dxdt_5 + 2*dxdt_6/55)
        xlist = np.concatenate((xlist,np.array([x])),axis = 0)

        i += 1
    
    return tlist, xlist

## implicit odesolvers
def odeii(fun,t_start: float,initial: np.ndarray,t_end: float,args:tuple=(),t_step: float = 1e-2,TOL: float = 1e-5,order:int = 4) -> tuple[np.ndarray,np.ndarray]:
    """ 
    implicit ode solver using BDFx method and will return time list with equal timestep

    return tlist, xlist 

    solutions of different time are stacked in axis = 0

    t_N = tlist[N,0] ; x_N = xlist[N,...]    
    ***************************************
    fun: fun(t,x,*args), return dx/dt function of the ode function to solve
    t_start: start of the 'time'
    initial: initial condition
    t_end: end of the time
    t_step: timestep for BDF solving
    order(int): order of BDF, range from 2 to 6
    """
    ## begin and validity check
    if t_start >= t_end:
        return np.array([t_start]), np.array([initial])
    if order > 6 or order < 2:
        raise ValueError('Supported BDFs\' order should be integer range from 2 to 6')

    ## start BDF
    # _, means there is a return value but we do not use it
    M_tlist, xlist = odeint(fun,t_start,initial,(t_start + (order)*t_step),args,t_step = t_step,step_max = t_step/10,TOL = TOL*0.01)
    M_tlist = M_tlist[0:order,...]
    xlist = xlist[0:order,...]
    M_xlist = np.copy(xlist)

    tlist = np.arange(t_start,t_end+t_step,t_step); N = np.size(tlist)
    match order:

        case 2:
            BDF = np.array([1,-4])/3
            LMM = t_step*np.array([3,-1])/2
            a = 2/3

        case 3:
            BDF = np.array([-2,9,-18])/11
            LMM = t_step*np.array([23,-16,5])/12
            a = 6/11
        
        case 4:
            BDF = np.array([3,-16,36,-48])/25
            LMM = t_step*np.array([-9,37,-59,55])/24
            a = 12/25
        
        case 5: 
            BDF = np.array([-12,75,-200,300,-300])/137
            LMM = t_step*np.array([1901, -2774, 2616, -1274, 251])/720
            a = 60/137

        case 6:
            BDF = np.array([10,-72,225,-400,450,-360])/147
            LMM = t_step*np.array([4277,-7923,9982,-7898,2877,-475])/1440
            a = 60/147

    shape = [order] + [1]*(xlist.ndim - 1)
    BDF = BDF.reshape(shape)
    eps = 1e-5
    i = order
    while(i < N):
        ## use BDF5 method to calculate 
        bias = np.sum(BDF*M_xlist,axis = 0)
        erf = 1
        ## use LMM4 to estimate a value 
        x = M_xlist[-1]

        for j in range(order):
            x = x + LMM[j]*fun(M_tlist[j],M_xlist[j],*args)

        M_tlist = M_tlist + t_step
        t = M_tlist[-1]
        while(erf >= TOL):
            F = x + bias - a*t_step*fun(t,x,*args)
            # ****** more accuracy
            #### dFdx = 1 - a*t_step*(16*fun(t,x+eps/2,*args) - fun(t,x+eps,*args) - 16*fun(t,x-eps/2,*args) + fun(t,x-eps,*args))/(14*eps) 
            dFdx = 1 - a*t_step*(fun(t,x+eps/2,*args) - fun(t,x-eps/2,*args) )/(eps) 
            x = x - F/dFdx

            erf = np.abs(np.max(F,axis = None))

        ## update Mlist
        xlist = np.concatenate((xlist,np.array([x])),axis = 0)
        M_xlist = np.roll(M_xlist,-1,axis = 0) #### very important
        M_xlist[-1,...] = x
        i += 1
      
    ## return
    return tlist,xlist

## tests
def ode00(fun,t_start: float,initial: np.ndarray,t_end: float,args:tuple=(),step_max: float = 1e-2,TOL: float = 1e-5) -> tuple[np.ndarray,np.ndarray]:
    """
    A test odesolver work for debugging

    return tlist, xlist 

    solutions of different time are stacked in axis = 0

    t_N = tlist[N,0] ; x_N = xlist[N,...]  

    Attention!!! 
    We do not recommend using this one to solve ode, but it may 
    help when debuging or testing, parameters set are the same as other odesolvers 
    """
## begin and validity check
    if t_start >= t_end:
        return np.array([t_start]), np.array([initial])

    t = t_start
    x = initial
    tlist = np.zeros([1,1])
    tlist[0,0] = t
    xlist = np.array([initial])

    ## operate
    while(t < t_end):
        ## ode body
        x = x + step_max*fun(t,x,*args)
        t += step_max

        tlist = np.vstack((tlist,t))
        xlist = np.concatenate((xlist,np.array([x])),axis = 0)
        
    ## return
    return tlist,xlist

def fun(t,x,w):
    dxdt = np.array([ x[1],-(w**2*x[0])])
    return dxdt

def ode_test(order):
    ## users set
    tspan = np.array([0,3])
    x0 = np.array([2,0])
    
    ## operate
    tlist,xlist = odeii(fun,0,x0,3,args = (3,),order = order,t_step = 0.06)
    ttlist,xxlist = ode23(fun,tspan[0],x0,tspan[1],args = (3,))
    
    ## figrue
    plt.figure(1)
    plt.plot(tlist,xlist[:,0],label = 'Position_ii')
    plt.plot(tlist,xlist[:,1],label = 'Velocity_ii')
    plt.plot(ttlist,xxlist[:,0],label = 'Position_23')
    plt.plot(ttlist,xxlist[:,1],label = 'Velocity_23')
    plt.legend()
    plt.show()

    plt.figure(3)
    nt = np.size(tlist)
    ntt = np.size(ttlist)
    nt_list = np.arange(1,nt+1,1)
    ntt_list = np.arange(1,ntt+1,1)
    plt.plot(nt_list,tlist,label = 'implicity')
    plt.plot(ntt_list,ttlist,label = 'explicity')
    plt.legend()
    plt.title('time step')
    plt.show()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    ord = 3
    ode_test(ord)
