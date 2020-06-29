import numpy as np

class Thruster_model(object):

    """
        Thruster model for spacecraft computes force and torque in the body frame and converts
        to inertial frame
        Commanded thrust is clipped to lie between zero and one, and then scaled based off of 
        thrust capability
        NOTE: maximum thrust is achieved with:
            tcmd = np.zeros(12)
            tcmd[0:2]=np.ones(2)
            tcmd[4:6]=np.ones(2)
            tcmd[8:10]=np.ones(2)
        and is norm([2,2,2])=3.46

    """
    def __init__(self,  pulsed=False, max_thrust=2.0, debug_fail=False, fail_idx_override=None):
        #                   dvec                      body position
        config = [
                      [ 1.0,  0.0,    0.0,   -1.0,    0.0,    0.0 ],  # upper -x face, rotate around y
                      [-1.0,  0.0,    0.0,    1.0,    0.0,    0.0 ],  # upper +x face, rotate around y

                      [ 0.0,  1.0,    0.0,   -0.0,   -1.0,    0.0 ],  # left -y face, rotate around z
                      [ 0.0, -1.0,    0.0,   -0.0,    1.0,    0.0 ],  # left +y face, rotate around z

                      [ 0.0,  0.0,    1.0,    0.0,   -0.0,   -1.0 ],  # left +z face, rotate around x
                      [ 0.0,  0.0,   -1.0,    0.0,   -0.0,    1.0 ]   # left -z face, rotate around x
                    ]

        config = np.asarray(config)

        self.dvec = config[:,0:3]

        self.position = config[:,3:6]

        self.num_thrusters = self.position.shape[0]
 
        self.max_thrust = np.ones(6)* max_thrust 

        self.reward_scale = np.sum(max_thrust)

        self.pulsed = pulsed 
   
        self.Isp = 210.0 
        self.g_o = 9.81
 
        self.eps = 1e-8

        self.mdot = None 

        self.debug_fail = debug_fail

        self.fail_idx_override = fail_idx_override

        self.fail_scale = None 
                             
        self.fail_idx = 0 

        self.fail = False

        print('thruster model: ')
          
    def thrust(self,commanded_thrust):

        assert commanded_thrust.shape[0] == self.num_thrusters

        if self.pulsed:
            commanded_thrust = commanded_thrust > self.eps
            #print('CM: ', commanded_thrust)

        if commanded_thrust[0] and commanded_thrust[1]:
            commanded_thrust[0] = commanded_thrust[1] = 0
        if commanded_thrust[2] and commanded_thrust[3]:
            commanded_thrust[2] = commanded_thrust[3] = 0
        if commanded_thrust[4] and commanded_thrust[5]:
            commanded_thrust[4] = commanded_thrust[5] = 0
            

        commanded_thrust = np.clip(commanded_thrust, 0, 1.0) * self.max_thrust

        
        if self.fail:
            if self.fail_idx_override is None:
                fail_idx = self.fail_idx
            else:
                fail_idx = self.fail_idx_override
   
            if self.debug_fail:
                orig_commanded_thrust = commanded_thrust.copy()
            commanded_thrust[fail_idx] = commanded_thrust[self.fail_idx] * self.fail_scale
            if self.debug_fail:
                if not np.all(orig_commanded_thrust ==  commanded_thrust):
                    print('orig: ', orig_commanded_thrust)
                    print('mod:  ', commanded_thrust) 
 
        force = np.expand_dims(commanded_thrust,axis=1) * self.dvec
 
        torque = np.cross(self.position, force)

        force = np.sum(force,axis=0)
        torque = np.sum(torque,axis=0)

        mdot = -np.sum(np.abs(commanded_thrust)) / (self.Isp * self.g_o)
        self.mdot = mdot # for rewards
        return commanded_thrust, force, torque, mdot
