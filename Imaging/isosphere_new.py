import numpy as np
import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
import pylab as pl
import scipy as sp

class Isosphere(object):

    def __init__(self, recursion_level=1):
        self.recursion_level = recursion_level
 
        self.reset()

    def show_stats(self):
        print('Sphere has %d faces and %d vertices' % (len(self.faces), len(self.vertices)))
        print('Min / Max Vertices: ' , np.min(self.vertices,axis=0),  np.max(self.vertices,axis=0))

    def reset(self):
        self.vertex_idx = 0
        self.vertices = []
        self.middle_point_cache = {}

        self.init_vertices()
        self.init_faces()

        for i in range(self.recursion_level):
            self.refine_triangles()
        
       
    def init_vertices(self):
        t = (1.0 + np.sqrt(5.0)) / 2.0

        self.add_vertex((-1,  t,  0))
        self.add_vertex(( 1,  t,  0))
        self.add_vertex((-1, -t,  0))
        self.add_vertex(( 1, -t,  0))

        self.add_vertex(( 0, -1,  t))
        self.add_vertex(( 0,  1,  t))
        self.add_vertex(( 0, -1, -t))
        self.add_vertex(( 0,  1, -t))

        self.add_vertex(( t,  0, -1))
        self.add_vertex(( t,  0,  1))
        self.add_vertex((-t,  0, -1))
        self.add_vertex((-t,  0,  1))
 
    def add_vertex(self,  location):
        location = location / np.linalg.norm(location) 
        self.vertices.append(location)
        vertex_idx = self.vertex_idx     
        self.vertex_idx += 1
        #print('debug: ', vertex_idx, self.vertex_idx)
        return vertex_idx

    def init_faces(self):
        self.faces = []

        #  5 faces around point 0
        self.faces.append(np.asarray([0, 11, 5]))
        self.faces.append(np.asarray([0, 5, 1]))
        self.faces.append(np.asarray([0, 1, 7]))
        self.faces.append(np.asarray([0, 7, 10]))
        self.faces.append(np.asarray([0, 10, 11]))

        #  5 adjacent faces 
        self.faces.append(np.asarray([1, 5, 9]))
        self.faces.append(np.asarray([5, 11, 4]))
        self.faces.append(np.asarray([11, 10, 2]))
        self.faces.append(np.asarray([10, 7, 6]))
        self.faces.append(np.asarray([7, 1, 8]))

        #  5 faces around point 3
        self.faces.append(np.asarray([3, 9, 4]))
        self.faces.append(np.asarray([3, 4, 2]))
        self.faces.append(np.asarray([3, 2, 6]))
        self.faces.append(np.asarray([3, 6, 8]))
        self.faces.append(np.asarray([3, 8, 9]))

        #  5 adjacent faces 
        self.faces.append(np.asarray([4, 9, 5]))
        self.faces.append(np.asarray([2, 4, 11]))
        self.faces.append(np.asarray([6, 2, 10]))
        self.faces.append(np.asarray([8, 6, 7]))
        self.faces.append(np.asarray([9, 8, 1]))
        

    def get_middle_point(self, p1, p2):

        if p1 > p2:
            key = str(p2) + '_' + str(p1)
        else:
            key = str(p1) + '_' + str(p2)
 
        if key in self.middle_point_cache:
            #print('found key, early return')
            return self.middle_point_cache[key]
         
        point1 = self.vertices[p1]
        point2 = self.vertices[p2]

        middle = (point1 + point2) / 2
        idx = self.add_vertex(middle)

        self.middle_point_cache[key] = idx

        return idx        
        
    def refine_triangles(self):
        faces2 = []
        for pnts in self.faces:
            a = self.get_middle_point(pnts[0] , pnts[1])
            b = self.get_middle_point(pnts[1] , pnts[2])
            c = self.get_middle_point(pnts[2] , pnts[0])

            faces2.append(np.asarray([pnts[0], a , c]))
            faces2.append(np.asarray([pnts[1], b , a]))
            faces2.append(np.asarray([pnts[2], c , b]))
            faces2.append(np.asarray([a , b , c]))

        self.faces = faces2
             

    def faces2vertices(self):
        V = []
        for f in self.faces:
            v = np.zeros((3,3))
            for i in range(len(f)):
                p = self.vertices[f[i]]
                v[i] = p
            V.append(v)
        return V

    def show(self, axis_limit=1):
        V = self.faces2vertices()
        ax = a3.Axes3D(pl.figure())
        for vtx in V:
            tri = a3.art3d.Poly3DCollection([vtx])
            tri.set_color(colors.rgb2hex(sp.rand(3)))
            tri.set_edgecolor('k')
            ax.add_collection3d(tri)
            ax.set_xlim(-axis_limit,axis_limit)
            ax.set_ylim(-axis_limit,axis_limit)
            ax.set_zlim(-axis_limit,axis_limit)
        pl.gca().set_zlabel('Z (m)')
        pl.gca().set_ylabel('Y (m)')
        pl.gca().set_xlabel("X (m)")
        pl.show()

    def get_triangle_vertices(self):
        v0 = []
        v1 = []
        v2 = []

        for face in self.faces:
            v0.append(self.vertices[face[0]])
            v1.append(self.vertices[face[1]])
            v2.append(self.vertices[face[2]])

        return v0, v1, v2
             
    def apply_shape_model(self, shape_model):
        sm_dvecs = shape_model / np.linalg.norm(shape_model)
        new_vertices = []
        for v in self.vertices:
            dots = np.sum(v * sm_dvecs,axis=1)
            hit_idx = np.argmax(dots)
            new_vertices.append(shape_model[hit_idx])
        self.vertices = new_vertices

                                       
    def perturb(self, p_scale=0.1, r_scale=500.):
        self.reset()
        vertices = np.vstack(self.vertices)
        p_vertices = vertices + np.random.uniform(low=-p_scale,high=p_scale,size=vertices.shape)
        p_vertices *= r_scale 
        self.vertices = list(p_vertices)

    def perturb_axes(self, p_scale=0.1, r_scale_p=(500.,500.,500.),  r_scale_n=(500.,500.,500.)):
        self.reset()
        #print(r_scale)
        vertices = np.vstack(self.vertices)
        p_vertices = vertices + np.random.uniform(low=-p_scale,high=p_scale,size=vertices.shape)
        for i in range(3):
            idx_p = np.where(p_vertices[:,i] >= 0)[0]
            idx_n = np.where(p_vertices[:,i] < 0)[0]
            p_vertices[idx_p,i] *= r_scale_p[i]
            p_vertices[idx_n,i] *= r_scale_n[i]

                

        self.vertices = list(p_vertices)
        #print('Min / Max Vertices: ' , np.min(self.vertices,axis=0),  np.max(self.vertices,axis=0))

    def get_limits(self):
        max_lim = np.max(self.vertices,axis=0)
        min_lim = np.min(self.vertices,axis=0)
        return min_lim, max_lim

    def roughen(self,p):
        vertices = np.asarray(self.vertices)
        vertices += p*np.random.rand(vertices.shape[0],3)
        self.vertices = vertices
        
    def save(self, fname):
        data = {}
        data['vertices'] = self.vertices
        data['faces'] = self.faces
        np.save(fname, data)

    def load(self, fname):
        data = np.load(fname).item()
        self.vertices = data['vertices']
        self.faces = data['faces']
        print('Sphere has %d faces and %d vertices' % (len(self.faces), len(self.vertices)))
        print('Min / Max Vertices: ' , np.min(self.vertices,axis=0),  np.max(self.vertices,axis=0))

