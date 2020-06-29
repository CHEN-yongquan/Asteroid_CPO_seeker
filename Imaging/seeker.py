import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import attitude_utils as attu
import  optics_utils as optu 
#from pylab import imshow
import env_utils as envu
class Seeker(object):

    """

   
    """
 
    def __init__(self, attitude_parameterization=attu.Quaternion_attitude, C_cb=None, r_cb=np.zeros(3), 
                  fov=np.pi/2,  debug=False, radome_slope_bounds=(0.0,0.0), range_bias=(0.0,0.0), range_noise=0.0, angle_noise=0.0):
        self.attitude_parameterization = attitude_parameterization 
        self.radome_slope_bounds = radome_slope_bounds

        if C_cb is None:
            self.C_cb = np.identity(3)
        else:
            self.C_cb = C_cb

        self.r_cb = r_cb

        self.fov = fov

        self.e_u = np.expand_dims(np.asarray([1., 0., 0.]), axis=1)

        self.e_v = np.expand_dims(np.asarray([0., 1., 0.]), axis=1)

        print('C_cb: ')
        print(self.C_cb)
       
        self.debug = debug
        self.range_bias = range_bias
        self.range_noise = range_noise
        self.angle_noise = angle_noise

    def reset(self):
        self.radome_slope = np.random.uniform(self.radome_slope_bounds[0], self.radome_slope_bounds[1], size=2)
        self.rand_range_bias = np.random.uniform(self.range_bias[0], self.range_bias[1])
         
    def get_seeker_angles(self, agent_location, agent_q, object_locations, object_intensities1):

        """
        agent_coords:    inertial frame coords of agent with camera
        object_coords:   array of inertial frame coords of object producing 
                            ray (camera) or hit by ray (Lidar)   
        q:          agent's attitude

        C_bn -> Convert from inertial to body frame 
        M_cb -> Convert from body frame to camera frame

        1st convert to inertial body centered reference frame
        then rotate onto body frame
        once in body frame, we can translate so are centered on where the camera is mounted
        then we rotate onto camera frame

        """

        object_intensities = self.range_distort(object_intensities1)

        if len(object_locations.shape) < 2:
            object_locations = np.expand_dims(object_locations,axis=0)
        if len(object_intensities.shape) < 2:
            object_intensities = np.expand_dims(object_intensities,axis=1)
        # translate to body-centered inertial frame
        object_locations = object_locations - agent_location

        C_bn = self.attitude_parameterization.q2dcm(agent_q)
        # rotate to body-centered body frame then translate to camera centered
        bf_coords = object_locations.dot(C_bn.T) - self.r_cb

        # rotate to camera-centered camera frame
        cf_coords = bf_coords.dot(self.C_cb.T)
            
        z = np.expand_dims(cf_coords[:,2],axis=1)
        idx = np.where(z > 0)[0]                    # can't look behind us
        cf_coords = cf_coords[idx]
        range_vals = object_intensities[idx]

        x = cf_coords[:,0]
        y = cf_coords[:,1]
        scale =  np.expand_dims(np.linalg.norm(cf_coords,axis=1), axis=1)
        los = cf_coords / scale 
        theta_u = np.arcsin(np.clip(los.dot(self.e_u), -1, 1))
        theta_v = np.arcsin(np.clip(los.dot(self.e_v), -1, 1))
        theta_u += theta_u * self.radome_slope[0] + np.clip(np.random.normal(loc=0.0, scale=self.angle_noise), -3*self.angle_noise, 3*self.angle_noise) 
        theta_v += theta_v * self.radome_slope[1] + np.clip(np.random.normal(loc=0.0, scale=self.angle_noise), -3*self.angle_noise, 3*self.angle_noise)
        if self.debug:
            print('THETA1: ', theta_u, theta_v)
        idx_u = np.abs(theta_u) <  self.fov/2
        idx_v = np.abs(theta_v) <  self.fov/2
 
        idx = np.logical_and(idx_u, idx_v)

        theta_u = theta_u[idx] / (self.fov/2) / 1.00
        theta_v = theta_v[idx] / (self.fov/2) / 1.00
        range_vals = np.expand_dims(range_vals[idx],axis=1)
        seeker_angles = np.stack((theta_u, theta_v),axis=1)

        if self.debug:
            print('Begin Debug .......')
            print('C_bn:')
            print(C_bn)
            print('C_cb:')
            print(self.C_cb)
            print('Agent Locataion: ',agent_location)
            print('Agent q: ' , agent_q) 
            print('Object Locations:')
            print(object_locations)
            print('bf_coords: ')
            print(bf_coords)
            print('cf_coords: ')
            print(cf_coords)
            print('z: ', z)
            print('Pixel locs: ')
            print(seeker_angles)
            print('los: ',los)
            print('losdoteu: ',  los.dot(self.e_u))
            print('losdotev: ', los.dot(self.e_v))
            print('idx_u/v: ',idx_u, idx_v, idx)
            print('theta2: ', theta_u, theta_v)
            print('Optical axis: ',self.get_optical_axis(C_bn))
            print('End Debug ........')
        return seeker_angles, range_vals 


    def render(self, agent_location, q, object_locations, object_intensities, default_intensity=0.0):
        pixels, intensities = self.get_pixel_coords(agent_location, q, object_locations, object_intensities)
        u = np.round(pixels[:,0]).astype(int) + self.p_x
        v = np.round(pixels[:,1]).astype(int) - self.p_y

        image = default_intensity*np.ones((self.p_x,self.p_y))
        image[v,u] = intensities
        plt.figure()
        plt.imshow(image, interpolation='nearest',cmap='gray')
        plt.grid(True)
 
 
    def getM_cb(self):
        return self.M_cb

    def getC_bn(self, q):
        C_bn = self.attitude_parameterization.q2dcm(q)
        return C_bn
       
    def getC_cn(self, q):
        C_cn = self.getM_cb.dot(self.getC_bn(q))
        return C_cn

    def get_bf_optical_axis(self):
        c_opt = 1.0*np.asarray([0,0,1])
        b_opt = self.C_cb.T.dot(c_opt)
        return(b_opt)

    def get_optical_axis(self, C_bn):
        b_opt = self.get_bf_optical_axis()
        nf_opt = C_bn.T.dot(b_opt)
        return nf_opt

    def get_optical_axis_q(self, q):
        C_bn = self.attitude_parameterization.q2dcm(q)        
        b_opt = self.get_bf_optical_axis()
        nf_opt = C_bn.T.dot(b_opt)
        return nf_opt

    def range_distort(self, r):
        rd = r + r * self.rand_range_bias + np.clip(np.random.normal(loc=0.0, scale=self.range_noise), -3*self.range_noise, 3*self.range_noise) 
        return rd
 
