import numpy as np
import attitude_utils as attu
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import env_utils as envu

class Sensor(object):
    def __init__(self, seeker, max_range_intensity=0.0, attitude_parameterization=attu.Quaternion_attitude,  use_range=True, 
                    pool_type='max', offset=np.asarray([0,0]), state_type=None, optflow_scale=1.0 ,
                    apf_tau1=300, apf_v0=0.5, use_dp=True, landing_site_range=0.0, debug=False):
        self.debug = debug
        self.ap =  attitude_parameterization 
        self.use_range = use_range
        self.stabilized = True
        self.landing_site_range = landing_site_range
        self.use_dp = use_dp
        self.seeker = seeker
        print(seeker.get_optical_axis(np.identity(3)))
        self.max_range_intensity = max_range_intensity
        self.seeker_angles = None
        self.pixel_int = None
        self.optflow_scale = optflow_scale

        self.apf_v0 = apf_v0
        self.apf_tau1 = apf_tau1

        self.track_func = self.track_func1

        #self.c_dvec = None
        self.offset = offset
        if pool_type == 'ave':
            self.pool_func = self.ave_pool_forward_reshape
            print('using average pooling')
        else:
            self.pool_func = self.max_pool_forward_reshape 
            print('using max  pooling')

        if state_type is None:
            self.state_type=Range_sensor.simple_state
        else:
            self.state_type = state_type 
        print('V4: Output State type: ', state_type)
            
    def reset(self, lander_state):
        self.seeker.reset()
        self.initial_attitude = lander_state['attitude'].copy()
        self.seeker_angles = None
        self.cs_angles = None
        self.pixel_int = None
        self.image_f = None
        self.image_c = None
        self.full_image = None
        self.last_seeker_angles = None
        self.last_pixel_int = None


    def get_seeker_angles(self, agent_state,  object_locations=np.zeros(3), render=False ):
        agent_location = agent_state['position']
        agent_velocity = agent_state['velocity']
        out_of_fov = False
        if len(object_locations.shape) < 2:
            object_locations = np.expand_dims(object_locations,axis=0)
        object_intensities = np.linalg.norm(agent_location-object_locations,axis=1)
        if self.stabilized:
            agent_q = self.initial_attitude
        else:
            agent_q = agent_state['attitude']
        self.agent_q = agent_q
        seeker_angles, pixel_int = self.seeker.get_seeker_angles(agent_location, agent_q, object_locations, object_intensities)
        if render:
            self.render(seeker_angles, pixel_int)
        #pixel_int = np.squeeze(pixel_int)


        #print('sensor: ', pixel_int, np.linalg.norm(agent_location))

        self.fov_violation =  seeker_angles.shape[0] < 1

        if seeker_angles.shape[0] < 1:
            seeker_angles = 1.0*np.expand_dims(1.1*self.seeker.fov/2*np.ones(2), axis=0)
        else:
            seeker_angles =  seeker_angles

        pixel_vc = envu.get_vc(agent_location, agent_velocity)

        return seeker_angles, pixel_int, pixel_vc

    def get_image_state(self, agent_state,  object_locations  ):

        agent_location = agent_state['position']
        agent_velocity = agent_state['velocity']

        seeker_angles,  pixel_int , pixel_vc = self.get_seeker_angles( agent_state,  object_locations=object_locations )

        seeker_angles = np.squeeze(seeker_angles)

        self.traj_seeker_angles = seeker_angles.copy()

        pixel_int = np.squeeze(pixel_int)
        self.pixel_int = pixel_int
        if self.fov_violation:
            du = 0.0
            dv = 0.0
        elif self.last_seeker_angles is not None:
            #print('PC2: ', seeker_angles, self.last_seeker_angles)
            du = 1.0*(seeker_angles[0] - self.last_seeker_angles[0])
            dv = 1.0*(seeker_angles[1] - self.last_seeker_angles[1])
        else:
            du = 0.0
            dv = 0.0

        du *= self.optflow_scale
        dv *= self.optflow_scale

        self.du = du
        self.dv = dv

        if self.fov_violation :
            pixel_int = 0.0

        if self.fov_violation :
            dp = 0.0
        elif self.last_seeker_angles is not None:
            dp = pixel_int - self.last_pixel_int
        else:
            dp = 0.0

        self.last_seeker_angles = seeker_angles.copy()
        self.last_pixel_int = pixel_int

        self.cs_angles =  seeker_angles - self.offset 
        self.seeker_angles = seeker_angles.copy()

        if self.use_dp:
            verr, t_go = self.track_func(pixel_int, dp)
        else:
            verr, t_go = self.track_func(pixel_int, pixel_vc)

        state = self.state_type( self.cs_angles, pixel_int, pixel_vc, du, dv, dp, verr, t_go)
        #if self.fov_violation:
        #    print(state)
        #    assert False

        self.verr = verr
        if self.debug and False:
            print('2:',seeker_angles, state, self.cs_angles * (self.seeker.p_y//2))
        return state

    @staticmethod
    def optflow_state(seeker_angles, pixel_int, pixel_vc, du, dv, dp, verr, t_go):
        state = np.hstack((seeker_angles,du,dv))
        return state

    @staticmethod
    def range_dp_state0(seeker_angles, pixel_int, pixel_vc, du, dv, dp, verr, t_go):
        state = verr
        #print(state)
        return state

    @staticmethod
    def range_dp_state1(seeker_angles, pixel_int, pixel_vc, du, dv, dp, verr, t_go):
        state = np.hstack((pixel_int, dp))
        #print(state)
        return state

    @staticmethod
    def range_dp_state2(seeker_angles, pixel_int, pixel_vc, du, dv, dp, verr, t_go):
        state = np.hstack((pixel_int, dp, t_go))
        #print(state)
        return state

    @staticmethod
    def optflow_state_range_dp00(seeker_angles, pixel_int, pixel_vc, du, dv, dp, verr, t_go):
        state = np.hstack((seeker_angles,du,dv,pixel_int, dp, t_go))
        return state

    @staticmethod
    def optflow_state_range_dp0(seeker_angles, pixel_int, pixel_vc, du, dv, dp, verr, t_go):
        state = np.hstack((seeker_angles,du,dv,pixel_int, dp))
        return state

    @staticmethod
    def optflow_state_range_dp1(seeker_angles, pixel_int, pixel_vc, du, dv, dp, verr, t_go):
        state = np.hstack((seeker_angles,du,dv,verr, t_go))
        return state

        state = np.hstack((seeker_angles,du,dv,verr, t_go))
        return state

    @staticmethod
    def optflow_state_range_dp2(seeker_angles, pixel_int, pixel_vc, du, dv, dp, verr, t_go):
        state = np.hstack((seeker_angles,du,dv,verr))
        return state

    @staticmethod
    def optflow_state_range_dp3(seeker_angles, pixel_int, pixel_vc, du, dv, dp, verr, t_go):
        state = np.hstack((seeker_angles,du,dv,verr))
        return state

    def check_for_vio(self):
        return self.fov_violation

 
    def render(self, pixels, intensities):
        u = pixels[:,0]
        v = pixels[:,1]

        image = self.max_range_intensity*np.ones((self.seeker.p_x,self.seeker.p_y))
        image[v,u] = intensities
        plt.figure()
        plt.imshow(image, interpolation='nearest',cmap='gray')
        plt.grid(True)

    def max_pool_forward_reshape(self, x, stride, pool_height, pool_width):
        """
        A fast implementation of the forward pass for the max pooling layer that uses
        some clever reshaping.

        This can only be used for square pooling regions that tile the input.
        """
        H, W = x.shape
    
        assert pool_height == pool_width == stride, 'Invalid pool params'
        assert H % pool_height == 0
        assert W % pool_height == 0
        x_reshaped = x.reshape(H // pool_height, pool_height,
                               W // pool_width, pool_width)
        out = x_reshaped.max(axis=1).max(axis=2)
        return out

    def ave_pool_forward_reshape(self, x, stride, pool_height, pool_width):
        """
        A fast implementation of the forward pass for the max pooling layer that uses
        some clever reshaping.

        This can only be used for square pooling regions that tile the input.
        """
        H, W = x.shape
    
        assert pool_height == pool_width == stride, 'Invalid pool params'
        assert H % pool_height == 0
        assert W % pool_height == 0
        x_reshaped = x.reshape(H // pool_height, pool_height,
                               W // pool_width, pool_width)
        out = x_reshaped.mean(axis=1).mean(axis=2)
        return out


    def track_func0(self, r, dr):

        if np.abs(dr) > 0:
            t_go = np.abs(r / dr)
        else:
            t_go = 9999

        vref = self.apf_v0 * (1. - np.exp(-t_go / self.apf_tau1))
        verr = dr - vref

        return verr, t_go

    def track_func1(self, r, dr):
        r -= self.landing_site_range
        if np.abs(dr) > 0:
            t_go = np.abs(r / dr)
        else:
            t_go = 9999

        if r < 0:
            t_go = 0.0

        vref = self.apf_v0 * (1. - np.exp(-t_go / self.apf_tau1))
        verr = dr - vref
        #print('track: ',vref, r, dr, t_go)
        return verr, t_go

 
