import numpy as np
import attitude_utils as attu

class Landing_icgen(object):

    def __init__(self, sc_range,  
                 attitude_parameterization=None,
                 mag_v =(0.05,0.10),
                 position_error=(0,np.pi/4),
                 heading_error=(0,np.pi/4),
                 attitude_error=(0,np.pi/16),
                 lander_wll=(0.0,0.0,0.0),
                 lander_wul=(0.0,0.0,0.0),
                 min_mass=450, max_mass=500,
                 p_engine_fail=0.0,
                 engine_fail_scale=(0.5,1.0),
                 debug_fail=False,
                 debug=False, 
                 noise_u=np.zeros(3), noise_sd=np.zeros(3),  l_offset=0.0,
                 position=None,
                 velocity=None,
                 attitude=None,
                 inertia_uncertainty_diag=0.0, inertia_uncertainty_offdiag=0.0):
        
        self.sc_range = sc_range
        self.mag_v = mag_v
        self.position_error = position_error
        self.heading_error = heading_error 

        self.attitude_parameterization=attitude_parameterization
        self.attitude_error=attitude_error
        self.lander_wul=lander_wul
        self.lander_wll=lander_wll

        self.debug_fail = debug_fail
        self.p_engine_fail = p_engine_fail
        self.engine_fail_scale = engine_fail_scale
 
        self.min_mass = min_mass
        self.max_mass = max_mass

        self.noise_u = noise_u
        self.noise_sd = noise_sd
        self.l_offset = l_offset

        self.inertia_uncertainty_diag = inertia_uncertainty_diag
        self.inertia_uncertainty_offdiag = inertia_uncertainty_offdiag 

        self.position = position
        self.velocity = velocity
        self.attitude = attitude

        self.max_pointing_error = 0.0    
        self.debug = debug


    def show(self):
        print('Landing_icgen:')
 
    def set_ic(self , lander, dynamics):
        lander.asteroid.reset()
        # ENGINE FAILURE
        assert lander.thruster_model.fail is not None 
        lander.thruster_model.fail = np.random.rand() < self.p_engine_fail
        lander.thruster_model.fail_idx =  np.random.randint(low=0,high=lander.thruster_model.num_thrusters)
        lander.thruster_model.fail_scale = np.random.uniform(low=self.engine_fail_scale[0], high=self.engine_fail_scale[1])
        if  self.debug_fail:
            print('Engine Fail? : ', self.p_engine_fail, lander.thruster_model.fail, lander.thruster_model.fail_idx, lander.thruster_model.fail_scale)

        dynamics.noise_u = np.random.uniform(low=-self.noise_u, high=self.noise_u,size=3)
        dynamics.noise_sd = self.noise_sd

        lander.init_mass = np.random.uniform(low=self.min_mass, high=self.max_mass)
         
        r  = np.random.uniform(low=self.sc_range[0],       high=self.sc_range[1])
        landing_site, lam, radius = lander.asteroid.generate_random_surface_position()
        #print(landing_site)
        lam, theta_debug = attu.make_random_heading_error(self.position_error, lam)
         
        pos = (np.linalg.norm(landing_site) + r) * lam 
  
        dvec_p = -lam

        mag_v = np.random.uniform(low=self.mag_v[0], high=self.mag_v[1])
        dvec_v, theta_debug = attu.make_random_heading_error(self.heading_error, dvec_p)
        vel = mag_v * dvec_v  
        lander.state['position'] = pos 
        lander.state['velocity'] = vel  

        lander.state['attitude'] = attu.make_random_attitude_error(self.attitude_parameterization, self.attitude_error, dvec_p, np.asarray([0.,0.,-1]))
        lander.state['attitude_321'] = self.attitude_parameterization.q2Euler321(lander.state['attitude'])
        lander.state['w'] = np.random.uniform(low=self.lander_wll, high=self.lander_wul, size=3)
 
        lander.state['thrust'] = np.zeros(3)
        lander.state['mass']   = lander.init_mass

        if self.position is not None:
            lander.state['position'] = self.position
        if self.velocity is not None:
            lander.state['velocity'] = self.velocity
        if self.attitude is not None:
            lander.state['attitude'] = self.attitude

        if self.debug:
            print('debug v: ', 180 / np.pi * np.arccos(np.clip(np.dot(dvec_v, dvec_p),-1,1)), 180 /  np.pi * theta_debug)
            C = self.attitude_parameterization.q2dcm(lander.state['attitude'])
            sensor = np.asarray([0.,0.,-1])
            rot_dvec = C.T.dot(sensor)
            error = 180/np.pi*np.arccos(np.clip(np.dot(rot_dvec,dvec_p),-1,1))
            self.max_pointing_error = np.maximum(error,self.max_pointing_error)
            print('debug attitude: ', error, self.max_pointing_error)

        it_noise1 = np.random.uniform(low=-self.inertia_uncertainty_offdiag, 
                                      high=self.inertia_uncertainty_offdiag, 
                                      size=(3,3))
        np.fill_diagonal(it_noise1,0.0)
        it_noise1 = (it_noise1 + it_noise1.T)/2
        it_noise2 = np.diag(np.random.uniform(low=-self.inertia_uncertainty_diag,
                            high=self.inertia_uncertainty_diag,
                            size=3))
        lander.inertia_tensor = lander.nominal_inertia_tensor + it_noise1 + it_noise2

    
        if self.debug and False:
            print(dynamics.g, lander.state['mass'])
            print(lander.inertia_tensor)
