import numpy as np

def rf_bisection(fun,a,b,TOL,args = (),N_max = 37,**kwargs):
    """
    Args:
        fun (_type_): _functions to find zeros_
        a (_type_): _description_
        b (_type_): _description_
        TOL (_type_): _description_
        args (_tuple_): args for fun, fun(x,*args)
    """
    A = fun(a,*args)
    B = fun(b,*args)
    if A*B > 0:
        raise ValueError("for bisection method f(a)*f(b) should be smaller than 0")
    if a >= b:
        raise ValueError(" b should be larger than a ")
        
    
    n = 0
    erf = 1
    while(abs(erf) > TOL and n < N_max):
        c = (a+b)/2
        erf = fun(c,*args)
        if erf*B < 0:
            a = c
            A = erf
        else:
            b = c
            B = erf
        n += 1   
    
    final = kwargs.get('final',True)  
    if final == True:
        c = rf_newton(fun,c,0,args,2)    
    return c

def rf_secant(fun,a,b,TOL,args = (),N_max = 37,**kwargs):
    erf = 1
    n = 0
    x0list = [a,b]
    while(abs(erf) > TOL and n < N_max):
        try:
            x0 = x0list[-1] - (x0list[-1] - x0list[-2])*fun(x0list[-1],*args)/(fun(x0list[-1],*args) - fun(x0list[-2],*args))
        except:
            x0 = rf_newton(fun,x0,TOL,args,N_max = N_max - n)
            x0list.append(x0)
            break
        erf = fun(x0,*args)
        x0list.append(x0)
        n += 1
    
    final = kwargs.get('final',False)
    if final == True:
        x0list[-1] = rf_newton(fun,x0list[-1],0,args,2)
    return x0list[-1]  
    
def rf_newton(fun,x0,TOL,args = (),N_max = 37):
    h = 1e-5
    erf = 1
    n = 0
    while(abs(erf) > TOL and n < N_max):
        #dfdx = ( - fun(x0+2*h,*args) + fun(x0-2*h,*args) + 16*( fun(x0+h,*args) - fun(x0-h,*args)) )/(28*h)
        dfdx = (fun(x0+h,*args) - fun(x0-h,*args))/(2*h)
        erf = fun(x0,*args)
        x0 = x0 - erf/dfdx
        n += 1
        
    return x0
    
def root_fingding(fun,a,b,TOL,args = (),N_max = 37,**kwargs):
    if a >= b:
        raise ValueError(" a should be smaller than b")
    
    if fun(a,*args)*fun(b,*args) < 0:
        erf = 1
        n = 0
        while(abs(erf) > TOL and n < N_max):
            n += 1
            x0 = rf_secant(fun,a,b,TOL,args,N_max = 1)
            c = (a + b)/2
            erf = fun(c,*args)
            f_x0 = fun(x0,*args)
            
            if x0 >= a and x0 <= b:
                if f_x0 * fun(b,*args) < 0:
                    if x0 >= c:
                        a = x0
                    else:
                        if erf*fun(b,*args) < 0:
                            a = c
                        else:
                            a = x0; b = c       
                else:
                    if x0 <= c:
                        b = x0
                    else:
                        if erf*fun(b,*args) > 0:
                            b = c
                        else:
                            a = x0; b = c  
            else:
                x0 = c
                if erf*fun(b,*args) < 0:
                    a = c
                else:
                    b = c           
     
    else:
        x0 = rf_secant(fun,a,b,TOL,args,N_max = N_max)    

    final = kwargs.get('final',True)
    if final == True:
        x0 = rf_newton(fun,x0,0,args,2)
    return x0   

def rf_test():
    def func(x):
        f =  x**3 - 2.2*x 
        return f
    
    g = lambda x: x*(x-1)*(x-2)*(x-3)*(x-4)
    
    x0 = root_fingding(g,2.1,3.2,TOL = 1e-5,N_max = 3,final = True)
    print(x0)
    print(g(x0))
    
if __name__ == "__main__":
    rf_test()