import numpy as np
import attitude_utils as attu
import itertools

def ct2m(C,T):
    """
    Converts:
        C:  DCM
        T:  translation
    to M, a combined homogonous coordinate matrix

    """

    M = np.zeros((4,4))
    M[0:3,0:3] = C
    M[0:3,3]   = -T
    M[3,3] = 1
    return M


def rotate_optical_axis(yaw, pitch, roll):

    """
    default is aligned with +z
   
    optaxis     yaw     pitch   roll
 
       X        pi/2    0.0     pi/2
      -Z        0.0     0.0     pi

    """

    ap = attu.Euler_attitude()
    q = np.asarray([yaw,pitch,roll])
    C_cb = ap.q2dcm(q)
    return C_cb
 
 
    
def make_cube(side, location):
    offset = np.asarray([-side/2,-side/2,-side/2]) + location
    object_locations = np.asarray(list(itertools.product([0,1], repeat=3))) * side + offset
    return object_locations

def make_grid(steps=4, bounds=-100, points=41):
    x = np.linspace(-bounds,bounds,points)
    y = np.linspace(-bounds,bounds,points)
    p = np.linspace(-steps,steps,2*steps+1)
    P = []
    for i in range(p.shape[0]): 
        line = np.stack((x,1.0*bounds/steps*p[i]*np.ones_like(x))).T
        #print('line: ', line)
        P.append(line.copy())
    for i in range(p.shape[0]):
        line = np.stack((1.0*bounds/steps*p[i]*np.ones_like(x),x)).T
        #print('line: ', line)
        P.append(line.copy())
    return np.vstack(P)
     
