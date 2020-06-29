import numpy as np
import attitude_utils as attu

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
    def __init__(self,  pulsed=False, max_thrust=2.0, debug_fail=False, fail_idx_override=None, com=np.zeros(3)):
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

        self.com = com

        self.dvec = config[:,0:3]

        self.position = config[:,3:6]

        self.num_thrusters = self.position.shape[0]
 
        self.max_thrust = np.ones(6)* max_thrust

        self.reward_scale = np.sum(max_thrust)

        self.reward_scale_att = 1

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

        #assert commanded_thrust.shape[0] == self.num_thrusters

        mag_tcom = commanded_thrust[0:6]

        theta_tcom =  commanded_thrust[6:12]
        phi_tcom   =  commanded_thrust[12:18]

        #print('before: ', theta_tcom)
        theta_tcom = theta_tcom / 10.
        theta_tcom = np.clip(theta_tcom , -0.09, 0.09)
        #print('after: ', theta_tcom)

        #print('before: ', phi_tcom)
        phi_tcom = np.pi * phi_tcom
        phi_tcom = np.clip(phi_tcom, -np.pi, np.pi)
        #print('after: ', phi_tcom)

        dvec = self.dvec.copy()
        for i in range(6):
            dvec[i] = self.rotate_thrust(theta_tcom[i], phi_tcom[i], self.dvec[i]) 

        #print('before: ', self.dvec)
        #print('after: ',  dvec) 
        if self.pulsed:
            mag_tcom = mag_tcom > self.eps
        mag_tcom = np.clip(mag_tcom, 0, 1.0) * self.max_thrust
        if self.fail:
            if self.fail_idx_override is None:
                fail_idx = self.fail_idx
            else:
                fail_idx = self.fail_idx_override
   
            if self.debug_fail:
                orig_mag_tcom = mag_tcom.copy()
            mag_tcom[fail_idx] = mag_tcom[self.fail_idx] * self.fail_scale
            if self.debug_fail:
                if not np.all(orig_mag_tcom ==  mag_tcom):
                    print('orig: ', orig_mag_tcom)
                    print('mod:  ', mag_tcom) 
 
        force = np.expand_dims(mag_tcom,axis=1) * dvec
 
        torque = np.cross(self.position + self.com, force)
        force = np.sum(force,axis=0)
        torque = np.sum(torque,axis=0)

        mdot = -np.sum(np.abs(commanded_thrust)) / (self.Isp * self.g_o)
        self.mdot = mdot # for rewards
        return commanded_thrust, force, torque, mdot


    def rotate_thrust(self, theta, phi, dvec):
        z = np.asarray([0.,0.,1.])
        C = attu.DCM3(dvec,z)
        rz = np.cos(theta)
        ry = np.sin(theta)*np.sin(phi)
        rx = np.sin(theta)*np.cos(phi)
        rot_dvec = np.asarray([rx,ry,rz])
        rot_dvec = C.T.dot(rot_dvec)
        return  rot_dvec
