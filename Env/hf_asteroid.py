import ellipsoid_gravity_utils as egu
import numpy as np
import env_utils as envu

class Asteroid(object):
    def __init__(self,
                    collect_stats=False,
                    landing_site_override=None,
                    axis_override=None,  
                    axis_range=(150,400), 
                    rho_range=(500,5000),       
                    min_srp=(-1e-6, -1e-6, -1e-6), 
                    max_srp=(1e-6, 1e-6, 1e-6),
                    nutation_range=(np.pi/4,np.pi/2),
                    omega_range=(1e-5,5e-4)):
                 
        self.collect_stats = collect_stats
        self.landing_site_override = landing_site_override 
        self.axis_override = axis_override 
        self.axis_range = axis_range
        self.rho_range= rho_range
        self.min_srp = min_srp
        self.max_srp = max_srp
        self.nutation_range= nutation_range
        self.omega_range = omega_range
       
        self.stat_keys = ['A','B', 'C', 'RHO', 'SIGMA' , 'THETA' , 'W_N' , 'PHI' , 'LZ']
        self.stat_formats = {}
        self.stat_formats['A'] = '{:12.0f}'
        self.stat_formats['B'] = '{:12.0f}'
        self.stat_formats['C'] = '{:12.0f}'
        self.stat_formats['LZ'] = '{:12.0f}'
        self.stat_formats['RHO'] = '{:12.1f}'
        self.stat_formats['SIGMA'] = '{:12.2f}'
        self.stat_formats['THETA'] = '{:12.2f}'
        self.stat_formats['W_N'] = '{:12.8f}'
        self.stat_formats['PHI'] = '{:12.4f}'
 
        self.stats = {}
        for k in self.stat_keys:
            self.stats[k] = []
 

    def reset(self):
        if self.axis_override is None:
            self.c_axis = np.random.uniform(low=self.axis_range[0], high=self.axis_range[1])
            self.b_axis = np.random.uniform(low=self.c_axis+1, high=np.maximum(self.c_axis+1,self.axis_range[1]))
            self.a_axis = np.random.uniform(low=self.b_axis+1, high=np.maximum(self.b_axis+1,self.axis_range[1]))
        else:
            self.c_axis = self.axis_override[2]
            self.b_axis = self.axis_override[1]
            self.a_axis = self.axis_override[0]

        assert self.a_axis > self.b_axis and self.b_axis > self.c_axis
       
        self.rho = np.random.uniform(low=self.rho_range[0], high=self.rho_range[1])

        self.srp = np.random.uniform(low=self.min_srp, high=self.max_srp, size=3)

        self.omega = np.random.uniform(low=self.omega_range[0], high=self.omega_range[1])
        self.theta = np.random.uniform(low=self.nutation_range[0], high=self.nutation_range[1])
        Jx = self.b_axis**2 + self.c_axis**2 
        Jz = self.a_axis**2 + self.b_axis**2
        sigma = (Jz - Jx) / Jx 
        self.w_n = sigma * self.omega * np.cos(self.theta)
        self.phi = np.random.uniform(-np.pi, np.pi)                 

        if self.collect_stats:
            self.stats['A'] = self.a_axis
            self.stats['B'] = self.b_axis
            self.stats['C'] = self.c_axis
            self.stats['RHO'] = self.rho
            self.stats['SIGMA'] = sigma
            self.stats['W_N'] = self.w_n
            self.stats['PHI'] = self.phi
 
    def get_g(self,  position):
        dV, lam = egu.ellipsoid_gravity_field(position,self.a_axis,self.b_axis,self.c_axis,self.rho)
        g = dV * position
        return g

    def get_rot_acc(self, t, position, velocity):
        w_x = self.omega * np.sin(self.theta) * np.cos(self.w_n * t + self.phi)
        w_y = self.omega * np.sin(self.theta) * np.sin(self.w_n * t + self.phi)    
        w_z = self.omega * np.cos(self.theta)
        w = np.asarray([w_x, w_y, w_z])
        self.w = w
        coriolis_acc = np.cross(2 * velocity , w )
        centrifugal_acc =  np.cross(np.cross(w, position), w)   
        a_rot = coriolis_acc + centrifugal_acc
        return a_rot


    def generate_random_surface_position(self):
        theta = np.random.uniform(-np.pi/2,np.pi/2)
        phi = np.random.uniform(-np.pi,np.pi)
        x = self.a_axis * np.cos(theta) * np.cos(phi)
        y = self.b_axis * np.cos(theta) * np.sin(phi)
        z = self.c_axis * np.sin(theta)
        if self.landing_site_override is not None:
            pos = np.asarray([x,y,z])
        else:
            pos = self.landing_site_override
        pos = np.asarray([x,y,z])
        radius =  np.linalg.norm(pos)
        dvec = pos / radius
        self.landing_site_pn = dvec
        self.landing_site = pos
        if self.collect_stats:
            self.stats['LZ'].append(self.landing_site.copy())
        return pos, dvec, radius 

    def show_stats(self):
        for k in self.stat_keys:
            f = self.stat_formats[k]
            v = self.stats[k]
            s = '%-8s' % (k)
            s += envu.print_vector(' |',np.min(v,axis=0),f)
            s += envu.print_vector(' |',np.max(v,axis=0),f)
            print(s)

