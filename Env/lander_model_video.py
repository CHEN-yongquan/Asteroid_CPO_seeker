import numpy as np
import env_utils as envu
import attitude_utils as attu

class Lander_model(object):

    def __init__(self, asteroid, thruster_model, pt_sensor=None, sensor=None, use_trajectory_list=False, attitude_parameterization=None, divert = (0.0,  0.0, 0.0), 
                 landing_site_range=0.0, com_range=(0.0, 0.0), h=2, d=2, w=2, init_mass=500., attitude_bias=0.0, omega_bias=0.0):  
        self.traj = {}
        self.asteroid = asteroid
        self.attitude_bias = attitude_bias
        self.omega_bias = omega_bias
        self.com_range = com_range
        self.landing_site_range = landing_site_range
        self.state_keys = ['position','velocity','thrust','bf_thrust',  'torque', 'attitude', 'attitude_321', 'w',  'mass']
        self.trajectory_list = []
        self.trajectory = {} 
        self.sensor_dvec = np.asarray([0.,0.,-1.])
        self.init_mass = init_mass
        self.nominal_mass = self.init_mass
        self.attitude_parameterization = attitude_parameterization
        self.thruster_model = thruster_model
        self.sensor = sensor
        self.pt_sensor = pt_sensor
        m = self.init_mass
        self.inertia_tensor = 1./12 * m * np.diag([h**2 + d**2 , w**2 + d**2, w**2 + h**2])
        print('Inertia Tensor: ',self.inertia_tensor)
        self.nominal_inertia_tensor = self.inertia_tensor.copy()

        self.divert = divert
        self.use_trajectory_list = use_trajectory_list        

        self.get_state_agent    = self.get_state_agent_sensor 

        self.trajectory = {}
        self.state = {}
        self.prev_state = {}
        print('Lander Model: ')



    def clear_trajectory(self):
        for k in self.get_engagement_keys():
            self.trajectory[k] = []
        self.target_attitude = self.state['attitude'].copy()
        self.thruster_model.com = np.random.uniform(low=self.com_range[0], high=self.com_range[1], size=3)
        self.rand_attitude_bias = np.random.uniform(-self.attitude_bias, self.attitude_bias, 4)
        self.rand_omega_bias = np.random.uniform(-self.omega_bias, self.omega_bias, 3)

    def update_trajectory(self, done, t):

        es = self.get_landing_state(t)
        for k,v in es.items():
            self.trajectory[k].append(v)
         
        if(done):
            if self.use_trajectory_list:
                self.trajectory_list.append(self.trajectory.copy())

    def get_state_agent_sensor(self,t):
        state = self.sensor.get_image_state(self.state, object_locations=self.asteroid.landing_site)
        image_tmp = np.zeros((1,2,4,4))  # placeholder
        return image_tmp, state

    def get_state_agent_sensor_att_w(self,t):
        image = self.sensor.get_image_state(self.state, object_locations=self.asteroid.landing_site)
        att_error =  self.attitude_parameterization.distance(self.state['attitude'], self.target_attitude)
        state = np.hstack((image, self.state['attitude'], self.state['w'], att_error))
        image_tmp = np.zeros((1,2,4,4))  # placeholder
        return image_tmp, state


    def get_state_agent_sensor_att_w2(self,t):
        image = self.sensor.get_image_state(self.state, object_locations=self.asteroid.landing_site)
        noisy_attitude = self.state['attitude'] + self.state['attitude'] * self.rand_attitude_bias
        noisy_omega = self.state['w'] + self.state['w'] * self.rand_omega_bias
        att_error =  self.attitude_parameterization.sub(noisy_attitude, self.target_attitude)
        state = np.hstack((image, att_error,  noisy_omega))
        image_tmp = np.zeros((1,2,4,4))  # placeholder
        return image_tmp, state



    def get_state_dynamics(self):
        state = np.hstack((self.state['position'], self.state['velocity'],  self.state['w'], self.state['mass'],  self.state['attitude']))
        return state
 
    def show_cum_stats(self):
        print('Cumulative Stats (mean,std,max,argmax)')
        stats = {}
        argmax_stats = {}
        keys = ['thrust','glideslope']
        formats = {'thrust' : '{:6.2f}', 'glideslope' : '{:6.3f}', 'sc_margin' :  '{:6.3f}'} 
        for k in keys:
            stats[k] = []
            argmax_stats[k] = []
        for traj in self.trajectory_list:
            for k in keys:
                v = traj[k]
                v = np.asarray(v)
                if len(v.shape) == 1:
                    v = np.expand_dims(v,axis=1)
                wc = np.max(np.linalg.norm(v,axis=1))
                argmax_stats[k].append(wc)
                stats[k].append(np.linalg.norm(v,axis=1))
                 
        for k in keys:
            f = formats[k]
            v = stats[k]
            v = np.concatenate(v)
            #v = np.asarray(v)
            s = '%-8s' % (k)
            #print('foo: ',k,v,v.shape)
            s += envu.print_vector(' |',np.mean(v),f)
            s += envu.print_vector(' |',np.std(v),f)
            s += envu.print_vector(' |',np.min(v),f)
            s += envu.print_vector(' |',np.max(v),f)
            argmax_v = np.asarray(argmax_stats[k])
            s += ' |%6d' % (np.argmax(argmax_v))
            print(s)

    def show_final_stats(self,type='final'):
        if type == 'final':
            print('Final Stats (mean,std,min,max)')
            idx = -1
        else:
            print('Initial Stats (mean,std,min,max)')
            idx = 0
 
        stats = {}
        keys = ['norm_vf', 'norm_rf', 'position', 'velocity', 'fuel', 'attitude_321', 'w', 'glideslope','good_landing']
        formats = { 'norm_rf' : '{:8.3f}', 'norm_vf' : '{:8.3f}', 'position' : '{:8.1f}' , 'velocity' : '{:8.3f}', 'fuel' : '{:6.4f}', 'attitude_321' : '{:8.3f}', 'w' : '{:8.4f}', 'glideslope' : '{:8.3f}', 'good_landing' : '{:8.4f}'}

        for k in keys:
            stats[k] = []
        for traj in self.trajectory_list:
            for k in keys:
                v = traj[k]
                if k == 'landing_reward':
                    print(k,v)
                v = np.asarray(v)
                if len(v.shape) == 1:
                    v = np.expand_dims(v,axis=1)
                stats[k].append(v[idx])

        for k in keys:
            f = formats[k]
            v = stats[k]
            s = '%-8s' % (k)
            s += envu.print_vector(' |',np.mean(v,axis=0),f)
            s += envu.print_vector(' |',np.std(v,axis=0),f)
            s += envu.print_vector(' |',np.min(v,axis=0),f)
            s += envu.print_vector(' |',np.max(v,axis=0),f)
            print(s)


    def show_episode(self, idx=0):
        
        traj = self.trajectory_list[idx]
        t = np.asarray(traj['time'])
        p = np.asarray(traj['position'])
        v = np.asarray(traj['velocity'])
        c = np.asarray(traj['thrust'])

        f1 = '{:8.1f}'
        f2 = '{:8.3f}' 
        for i in range(t.shape[0]):

            s = 't: %6.1f' % (t[i])
            s += envu.print_vector(' |',p[i],f1)
            s += envu.print_vector(' |',v[i],f2)
            s += envu.print_vector(' |',c[i],f2)
            print(s)

         
    def get_landing_state(self,t):

        landing_state = {}
        landing_state['t'] = t
        landing_state['position'] = self.state['position'] 
        landing_state['velocity'] = self.state['velocity'] 
        landing_state['norm_rf'] = np.abs(np.linalg.norm(self.state['position']-self.asteroid.landing_site) - self.landing_site_range)
        landing_state['norm_vf'] = np.linalg.norm(self.state['velocity'])
        landing_state['v_ratio'] = np.linalg.norm(self.state['velocity'][0:2]) / np.abs(self.state['velocity'][2])
        landing_state['attitude'] = self.state['attitude']
        landing_state['attitude_321'] = self.state['attitude_321']
        landing_state['w']      =  self.state['w']
        landing_state['thrust'] = self.state['thrust'] 
        landing_state['mass']   = self.state['mass']
        landing_state['fuel']   = self.init_mass-self.state['mass']
        landing_state['vc'] =  envu.get_vc(self.state['position'], self.state['velocity']) 
        
        landing_state['seeker_angles'] = self.sensor.seeker_angles
        landing_state['cs_angles'] = self.sensor.cs_angles
        landing_state['optical_flow'] = np.hstack((self.sensor.du, self.sensor.dv))
        landing_state['v_err'] = self.sensor.verr
        C_bn = self.attitude_parameterization.q2dcm(self.state['attitude'])
        opt_axis = self.sensor.seeker.get_optical_axis(C_bn)
        v_dvec =  self.state['velocity'] / np.linalg.norm(self.state['velocity'])
        theta_cv = np.arccos(np.clip(np.dot(v_dvec,opt_axis), -1, 1))
        landing_state['theta_cv'] = theta_cv
        landing_state['good_landing'] = landing_state['norm_rf'] < 1.0 and landing_state['norm_vf'] < 0.1 
        landing_state['asteroid_w'] = self.asteroid.w
        return landing_state

    def get_engagement_keys(self):
        keys = ['t', 'norm_rf', 'norm_vf', 'position', 'velocity', 'attitude', 'attitude_321', 'w', 'thrust', 'bf_thrust', 'torque', 'mass', 'fuel', 'v_ratio','vc', 'reward','fuel_reward','tracking_reward','glideslope_reward', 'landing_margin', 'glideslope', 'sc_margin', 'glideslope_penalty','rh_penalty', 'sc_penalty','sc_reward','att_penalty','att_reward','landing_reward','value','w_reward','w_penalty',  'seeker_angles', 'cs_angles', 'optical_flow', 'theta_cv','fov_penalty','v_err','good_landing','asteroid_w']
        return keys


    def show(self):

        """
            For debugging

        """

        




 
