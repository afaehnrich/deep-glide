'''
======================
3D surface (color map)
======================

Demonstrates plotting a 3D surface colored with the coolwarm color map.
The surface is made opaque by using antialiased=False.

Also demonstrates using the LinearLocator and custom formatting for the
z axis tick labels.
'''

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np



from array import array

from traits.traits import Color

class TerrainClass:
    row_length = 3601
    resolution = 30 # in m
    
    def __init__(self):
        f = open('./SRTM/N44E006.hgt', 'rb')
        format = 'h'
        

        data = array(format)
        data.fromfile(f, self.row_length*self.row_length)
        data.byteswap()
        f.close()
        self.data = np.array(data).reshape(self.row_length,self.row_length)        

    def define_map(self, xdim, ydim, offset):
        dx = xdim // self.resolution
        dy = ydim // self.resolution
        self.xrange = (-dx*self.resolution/2, dx*self.resolution/2)
        self.yrange = (-dy*self.resolution/2, dy*self.resolution/2)
        #X=np.arange(self.xrange[0], self.xrange[1]+self.resolution*2, self.resolution)
        #Y=np.arange(self.yrange[0], self.yrange[1]+self.resolution*2, self.resolution)
        #self.X, self.Y = np.meshgrid(X, Y)
        self.X, self.Y = np.mgrid[  self.xrange[0]:self.xrange[1]+self.resolution*2:self.resolution, 
                                    self.yrange[0]:self.yrange[1]+self.resolution*2:self.resolution]
        self.Z = self.data[offset[0]:offset[0]+dx+2, offset[1]:offset[1]+dy+2]
        
    def altitude(self, x,y):
        id_x = int((x - self.xrange[0]) // self.resolution)
        x_rest = (x - self.xrange[0]) / self.resolution - id_x 
        id_y = int((y - self.yrange[0]) // self.resolution)
        y_rest = (y - self.yrange[0]) / self.resolution - id_y
        P0=np.array([id_x,id_y,self.Z[id_y, id_x]])
        Vx = np.array([1,0,self.Z[id_y,id_x+1]-P0[2]])
        Vy = np.array([0,1,self.Z[id_y+1,id_x]-P0[2]])
        P1 = P0 + Vx * x_rest + Vy * y_rest
        #print(P1[2],P0, Vx, x_rest, Vy, y_rest)
        return P1[2]
        

fig = plt.figure()
ax = fig.gca(projection='3d')



ter = TerrainClass()

import numpy
from mayavi import mlab

def test_plot3d():
    """Generates a pretty set of lines."""
    n_mer, n_long = 6, 11
    dphi = np.pi / 1000.0
    phi = np.arange(0.0, 2 * np.pi + 0.5 * dphi, dphi)
    mu = phi * n_mer
    x = np.cos(mu) * (1 + np.cos(n_long * mu / n_mer) * 0.5)
    y = np.sin(mu) * (1 + np.cos(n_long * mu / n_mer) * 0.5)
    z = np.sin(n_long * mu / n_mer) * 0.5

    l = mlab.plot3d(x, y, z, np.sin(mu), tube_radius=0.025, colormap='Spectral')
    return l

def test_surf():
    """Test surf on regularly spaced co-ordinates like MayaVi."""
    def f(x, y):
        sin, cos = np.sin, np.cos
        return sin(x + y) + sin(2 * x - y) + cos(3 * x + 4 * y)

    x, y = np.mgrid[-7.:7.05:0.1, -5.:5.05:0.05]
    print(x)
    s = surf(x, y, f)
    #cs = contour_surf(x, y, f, contour_z=0)
    return s

def test_points3d():
    t = np.linspace(0, 4 * np.pi, 20)

    x = np.sin(2 * t)
    y = np.cos(t)
    z = np.cos(2 * t)
    s = 2 + np.sin(t)

    return mlab.points3d(x, y, z, s, colormap="copper", scale_factor=.25)

# Make data.
ter.define_map(21000, 21000, [200,200])
X = ter.X
Y = ter.Y
Z = ter.Z
print(ter.altitude(-10500,10500))
# Plot the surface.
#s = surf(X,Y,Z)


r, theta = np.mgrid[0:10, -np.pi:np.pi:10j]

x = r * np.cos(theta)

y = r * np.sin(theta)

z = np.sin(r)/r


#mlab.mesh(x, y, z, colormap='gist_earth', extent=[0, 1, 0, 1, 0, 1])

m = mlab.mesh(x, y, z, extent=[0, 1, 0, 1, 0, 1], representation='wireframe', line_width=1, color=(0.5, 0.5, 0.5))
#mlab.colorbar(m, orientation='vertical')

#mlab.title('polar mesh')
#mlab.outline(m)
#mlab.axes(m)

#p = mlab.points3d([4000], [4000], [2000],1000)

p1 = mlab.points3d([1000], [-2000], [4000], scale_factor=400, color=(1,0,0))
l = mlab.plot3d([1000,2000,0],[-2000,3000,5000],[4000,3000,1000], line_width = 400., tube_radius=40, color=(0,0,1))
p2 = mlab.points3d([2000], [3000], [3000], scale_factor=400, color=(0,1,0))
surf = ax.plot_surface(X, Y, Z,
                       linewidth=1, alpha=0.5, antialiased=False)
s = mlab.surf(X,Y,Z)
l2 = mlab.plot3d([-1000,-2000,0],[2000,-3000,-5000],[-4000,-3000,1000], line_width = 400., tube_radius=40, color=(0,0,1))

input()
exit()
#ax.set_box_aspect((np.ptp(X), np.ptp(Y), np.ptp(Z)))
# Customize the z axis.
#ax.set_xlim(-21000/2, 21000/2)
#ax.set_ylim(-21000/2, 21000/2)
#ax.set_zlim(0, 10000)
np.random.seed()
# Add a color bar which maps values to colors.
for i in range(1, 200):
    x = np.random.uniform(-10500., 10500.)
    y = np.random.uniform(-10500., 10500.)
    z = ter.altitude(x,y)
    plt.plot(x,y,z,'r.')

plt.show()