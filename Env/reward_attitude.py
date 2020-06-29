import numpy as np
import env_utils as envu

class Reward(object):

    """
        Minimizes Velocity Field Tracking Error

    """

    def __init__(self,  reward_scale=1.0, ts_coeff=-0.0,  fuel_coeff=-0.01, landing_coeff = 10.0, 
                 landing_rlimit=1.0, landing_vlimit=0.2, landing_alimit=2*np.pi, landing_wlimit=0.2, landing_gslimit=-1,
                 tracking_coeff=-1.00, magv_coeff=-0.5, tracking_bias=0.01, debug=False, 
                 optflow_coeff=0.0, fov_coeff=0.0, fov_alt=5.0,  att_coeff=-0.0 ):

        self.reward_scale =         reward_scale
        self.ts_coeff =             ts_coeff
        self.fuel_coeff =           fuel_coeff
        self.att_coeff =            att_coeff

        self.landing_rlimit =       landing_rlimit
        self.landing_vlimit =       landing_vlimit
        self.landing_alimit =       landing_alimit
        self.landing_wlimit =       landing_wlimit
        self.landing_gslimit =      landing_gslimit

        self.magv_coeff =           magv_coeff

        self.fov_alt =              fov_alt
        self.fov_coeff =            fov_coeff
        self.optflow_coeff =        optflow_coeff

        self.landing_coeff =        landing_coeff

        self.tracking_coeff =       tracking_coeff
        self.tracking_bias =        tracking_bias


        self.debug =                debug

        print('Reward_terminal equator')

    def get(self, lander,  action, done, steps,  glideslope_constraint, attitude_constraint, w_constraint, rh_constraint):
        pos         =  lander.state['position'] - lander.asteroid.landing_site
        vel         =  lander.state['velocity']

        #state = np.hstack((pos,vel))

        r_gs = glideslope_constraint.get_reward()

        fov_penalty = 0.0
        rh_penalty = 0.0
        seeker_angles = lander.sensor.cs_angles

        optflow = np.asarray([lander.sensor.du,lander.sensor.dv])
        optflow_error = np.linalg.norm(optflow)

        alt = np.dot(lander.asteroid.landing_site_pn, pos)
 
        if alt > 0 and not lander.sensor.check_for_vio():
            tracking_error = np.linalg.norm(seeker_angles)
            tracking_error = self.tracking_coeff * tracking_error
            magv_error = self.magv_coeff * np.abs(lander.sensor.verr)
            att_error = self.att_coeff * lander.attitude_parameterization.distance(lander.state['attitude'], lander.target_attitude)
            r_tracking = self.tracking_bias + magv_error + tracking_error + att_error  
        else:
            r_tracking = 0.0

        r_att = attitude_constraint.get_reward(lander.state)
        r_w   = w_constraint.get_reward(lander.state)

        landing_margin = 0.
        gs_penalty = 0.0
        att_penalty = 0.0
        w_penalty = 0.0
        rh_penalty = 0.0
        r_landing = 0.0
        if done:

            if lander.sensor.check_for_vio() and alt > self.fov_alt:
                fov_penalty = self.fov_coeff

            gs_penalty = glideslope_constraint.get_term_reward()

            att_penalty = attitude_constraint.get_term_reward(lander.state)

            w_penalty = w_constraint.get_term_reward(lander.state)

            rh_penalty = rh_constraint.get_term_reward(lander.state)


            gs = glideslope_constraint.get()
            att = np.abs(lander.state['attitude_321'][1:3])
            w   = np.abs(lander.state['w'])
            margins = [np.linalg.norm(pos) -  self.landing_rlimit , np.linalg.norm(vel) -  self.landing_vlimit, np.max(np.abs(att)) -  self.landing_alimit, np.max(np.abs(w)) -  self.landing_wlimit,  self.landing_gslimit  - gs]
            landing_margin = np.max(margins)

            if np.linalg.norm(pos) < self.landing_rlimit and np.linalg.norm(vel) < self.landing_vlimit and np.max(w) < self.landing_wlimit:
                r_landing = self.landing_coeff

        reward_info = {}
        r_fuel = self.fuel_coeff * np.sum(np.abs(lander.state['bf_thrust'])) / (lander.thruster_model.reward_scale)

        reward_info['fuel'] = r_fuel

        reward1 = (r_w + fov_penalty + rh_penalty +  w_penalty + r_att +  att_penalty +  gs_penalty + r_gs + r_tracking +  r_fuel + self.ts_coeff) * self.reward_scale
        reward2 = r_landing * self.reward_scale

        reward = (reward1, reward2)
        lander.trajectory['reward'].append(reward1 + reward2)
        lander.trajectory['landing_reward'].append(r_landing)
        lander.trajectory['glideslope'].append(glideslope_constraint.get())
        lander.trajectory['glideslope_reward'].append(r_gs)
        lander.trajectory['glideslope_penalty'].append(gs_penalty)
        lander.trajectory['att_reward'].append(r_att)
        lander.trajectory['att_penalty'].append(att_penalty)
        lander.trajectory['w_reward'].append(r_w)
        lander.trajectory['w_penalty'].append(w_penalty)
        lander.trajectory['rh_penalty'].append(rh_penalty)
        lander.trajectory['fov_penalty'].append(fov_penalty)
        lander.trajectory['tracking_reward'].append(r_tracking)
        lander.trajectory['landing_margin'].append(landing_margin)
        lander.trajectory['fuel_reward'].append(r_fuel)
        return reward, reward_info


