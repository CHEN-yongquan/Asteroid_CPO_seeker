import ellipsoid_gravity_utils as egu
import numpy as np
import env_utils as envu

class Asteroid(object):
    def __init__(self,
                    collect_stats=False,
                    landing_site_override=None,
                    radius=250,
                    M_range=(2e10,20e10),
                    min_omega=(-1e-4, -1e-4, -1e-4), 
                    max_omega=(1e-4, 1e-4, 1e-4),
                    min_srp=(-1e-6, -1e-6, -1e-6), 
                    max_srp=(1e-6, 1e-6, 1e-6)):
              
        self.collect_stats = collect_stats 
        self.landing_site_override = landing_site_override  
        self.radius = radius
        self.min_omega = min_omega
        self.max_omega = max_omega 
        self.M_range=(2e10,20e10) 
        self.min_srp = min_srp
        self.max_srp = max_srp
        self.G = 6.674e-11
        self.stat_keys = ['M','W', 'LZ']
        self.stat_formats = {}
        self.stat_formats['M'] = '{:12.4f}'   
        self.stat_formats['W'] = '{:12.4f}' 
        self.stat_formats['LZ'] = '{:12.0f}'
        self.stats = {}
        for k in self.stat_keys:
            self.stats[k] = []

    def reset(self):
        self.M = np.random.uniform(low=self.M_range[0], high=self.M_range[1])
        self.w = np.random.uniform(low=self.min_omega, high=self.max_omega, size=3)
        self.srp = np.random.uniform(low=self.min_srp, high=self.max_srp, size=3)
        if self.collect_stats:
            self.stats['M'].append(self.M / 1e10)
            self.stats['W'].append(self.w * 1000)
 
    def get_g(self,  position):
        g = -self.G * self.M * position / np.linalg.norm(position)**3
        return g

    def get_rot_acc(self, t, position, velocity):
        coriolis_acc = np.cross(2 * velocity , self.w )
        centrifugal_acc =  np.cross(np.cross(self.w, position), self.w)   
        a_rot = coriolis_acc + centrifugal_acc
        return a_rot

    def generate_random_surface_position(self):
        theta = np.random.uniform(-np.pi/2,np.pi/2)
        phi = np.random.uniform(-np.pi,np.pi)
        x = self.radius * np.cos(theta) * np.cos(phi)
        y = self.radius * np.cos(theta) * np.sin(phi)
        z = self.radius * np.sin(theta)
        if self.landing_site_override is None:
            pos = np.asarray([x,y,z])
        else:
            pos = self.landing_site_override
        dvec = pos / self.radius
        self.landing_site_pn = dvec
        self.landing_site = pos
        #print(self.landing_site)
        if self.collect_stats:
            self.stats['LZ'].append(self.landing_site.copy())
        return pos, dvec, self.radius

    def show_stats(self):
        for k in self.stat_keys:
            f = self.stat_formats[k]
            v = self.stats[k]
            s = '%-8s' % (k)
            s += envu.print_vector(' |',np.min(v,axis=0),f)
            s += envu.print_vector(' |',np.max(v,axis=0),f)
            print(s)



