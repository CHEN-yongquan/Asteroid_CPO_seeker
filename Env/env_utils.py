import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def rad2deg(x):
    return 180/np.pi*x

def deg2rad(x):
    return np.pi/180*x


def get_vc(r_tm, v_tm):
   vc = -r_tm.dot(v_tm)/np.linalg.norm(r_tm)
   return vc

 
def print_vector(s,v,f):
    v = 1.0 * v
    if isinstance(v,float): 
         v = [v]
    s1 = ''.join(f.format(v) for k,v in enumerate(v))
    s1 = s + s1
    return s1


def rk4(t, x, xdot, h ):

    """
        t  :  time
        x  :  initial state
        xdot: a function xdot=f(t,x, ...)
        h  : step size

    """

    k1 = h * xdot(t,x)
    k2 = h * xdot(t+h/2 , x + k1/2)
    k3 = h * xdot(t+h/2,  x + k2/2)
    k4 = h * xdot(t+h   ,  x +  k3)

    x = x + (k1 + 2*k2 + 2*k3 + k4) / 6

    return x


