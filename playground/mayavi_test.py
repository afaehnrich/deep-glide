import numpy as np
from mayavi import mlab
from deep_glide.jsbgym_new.sim import TerrainOcean
from deep_glide.rrt_utils import plotting
import collections
import math
import numpy as np



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

# test_plot3d()
# terrain = TerrainOcean()
# terrain.define_map(4000,4000, (100, 100))
# flightRenderer = plotting.Plotting(None, None, None, terrain, mlab)
#         #self.flightRenderer.plot_start(self.start)
#         #self.flightRenderer.plot_goal(self.goal, 500)
#         #self.flightRenderer.plot_path(self.trajectory, radius=10)
#         #self.gui.process_events()
# input()

class BoundedProperty(collections.namedtuple('BoundedProperty', ['min', 'max', 'get'])):
    bound = True

class Property:
    bound = False

def PropertylistToBox(plist): #low=True
    boxLow = np.array([p.min for p in plist])
    boxHigh = np.array([p.max for p in plist])
    return boxLow, boxHigh

class Sim:
    def sim(self,d):
        print('simulating', d)
        return 2

class Test:
    
    sim: Sim = Sim()

    
    def call(self):
        print(self.prop)
        x = self.prop.get(self, 1)
        print('got ',x)

class Properties:
    prop = BoundedProperty(-1, 1, Test.sim.sim)



t = Test()
t.sim = Sim()
t.call()
