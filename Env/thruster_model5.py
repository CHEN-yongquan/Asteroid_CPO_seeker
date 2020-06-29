import numpy as np

class Thruster_model(object):

    """
        Thruster model for spacecraft computes force and torque in the body frame and converts
        to inertial frame
        Commanded thrust is clipped to lie between zero and one, and then scaled based off of 
        thrust capability
    """
    def __init__(self,  pulsed=False, debug_fail=False, fail_idx_override=None, max_thrust_att=0.10, max_thrust_trans=2.0, com=np.zeros(3)):
        #                   dvec                      body position
        #                   dvec                      body position
        config = [
                      [ 1.0,  0.0,    0.0,   -1.0,    0.0,    0.0 ],  # center -x face                      0
                      [-1.0,  0.0,    0.0,    1.0,    0.0,    0.0 ],  # center +x face,                     1

                      [ 0.0,  1.0,    0.0,   -0.0,   -1.0,    0.0 ],  # center -y face,                     2
                      [ 0.0, -1.0,    0.0,   -0.0,    1.0,    0.0 ],  # center +y face,                     3

                      [ 0.0,  0.0,    1.0,    0.0,   -0.0,   -1.0 ],  # left +z face,                       4
                      [ 0.0,  0.0,   -1.0,    0.0,   -0.0,    1.0 ],  # left -z face,                       5

                      [ 1.0,  0.0,    0.0,   -1.0,    0.0,    0.8 ],  # upper -x face, rotate around y      6
                      [-1.0,  0.0,    0.0,    1.0,    0.0,   -0.8 ],  # lower +x face, rotate around y      7

                      [ 1.0,  0.0,    0.0,   -1.0,    0.0,   -0.8 ],  # lower -x face, rotate around y      8
                      [-1.0,  0.0,    0.0,    1.0,    0.0,    0.8 ],  # upper +x face, rotate around y      9

                      [ 0.0,  1.0,    0.0,   -0.8,   -1.0,    0.0 ],  # left -y face, rotate around z      10
                      [ 0.0, -1.0,    0.0,    0.8,    1.0,    0.0 ],  # right +y face, rotate around z     11

                      [ 0.0,  1.0,    0.0,    0.8,   -1.0,    0.0 ],  # right -y face, rotate around z     12
                      [ 0.0, -1.0,    0.0,   -0.8,    1.0,    0.0 ],  # left +y face, rotate around z      13

                      [ 0.0,  0.0,    1.0,    0.0,   -0.8,   -1.0 ],  # left +z face, rotate around x      14
                      [ 0.0,  0.0,   -1.0,    0.0,    0.8,    1.0 ],  # right -z face, rotate around x     15

                      [ 0.0,  0.0,    1.0,    0.0,    0.8,   -1.0 ],  # right +z face, rotate around x     16
                      [ 0.0,  0.0,   -1.0,    0.0,   -0.8,    1.0 ]   # left -z face, rotate around x      17
                 ]

        trans_thrust = max_thrust_trans*np.ones(6)
        rot_thrust = max_thrust_att*np.ones(12)
        self.max_thrust1 = np.hstack((trans_thrust,rot_thrust))
        self.reward_scale = np.sum(self.max_thrust1)
        self.com = com

        #self.max_thrust = 2.0

        config = np.asarray(config)

        self.dvec = config[:,0:3]

        self.position = config[:,3:6]

        self.num_thrusters = self.position.shape[0]
 

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
                 
     
    def thrust(self,commanded_thrust):

        assert commanded_thrust.shape[0] == self.num_thrusters

        if self.pulsed:
            commanded_thrust = commanded_thrust > self.eps

        commanded_thrust = np.clip(commanded_thrust, 0, 1.0) * self.max_thrust1

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
 
        torque = np.cross(self.position + self.com, force)

        force = np.sum(force,axis=0)
        torque = np.sum(torque,axis=0)
        #print('F: ', force)
        mdot = -np.sum(np.abs(commanded_thrust)) / (self.Isp * self.g_o)
        self.mdot = mdot # for rewards
        return commanded_thrust, force, torque, mdot
