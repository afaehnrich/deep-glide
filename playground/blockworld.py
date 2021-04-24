from abc import abstractclassmethod, abstractmethod
from deep_glide.utils import elevation_asc2hgt
import jsbsim
import os
from typing import Dict, List
import time
import numpy as np
from array import array
import random
from deep_glide import plotting

from matplotlib import pyplot as plt


class TerrainClass:
    
    map_offset: List = None

    @abstractmethod
    def define_map_for_plotting(self, xrange, yrange):
        pass
        
    @abstractmethod
    def altitude(self, x,y):
        pass

    @abstractmethod
    def max_altitude(self, x, y, radius):
        pass

    @abstractmethod
    def map_around_position(self, x, y, width, height):
        pass

    @abstractmethod
    def get_map(self, p1, p2):
        pass

    @abstractmethod
    def pixel_from_coordinate(self, point):
        pass
  

class TerrainClass30m(TerrainClass):
    row_length = 3601
    resolution = 30 # in m
    filename = 'SRTM/30m/N46E008.hgt' # Schweiz
    
    def __init__(self):
        path = os.path.dirname(os.path.realpath(__file__))
        fname = os.path.join(path,self.filename) # Schweiz
        f = open(fname, 'rb')
        format = 'h'    
        data = array(format)
        data.fromfile(f, self.row_length*self.row_length)
        data.byteswap()
        f.close()
        self.data = np.array(data).reshape(self.row_length,self.row_length)       
        self.map_offset =  [self.row_length//2, self.row_length//2]

    def define_map_for_plotting(self, xrange, yrange):
        # TODO: Namen der Funktion ändern und nach plotting verschieben
        self.xrange = np.array(xrange, dtype=int) // self.resolution
        self.yrange = np.array(yrange, dtype=int) // self.resolution
        self.X, self.Y = np.mgrid[xrange[0]:xrange[1]:self.resolution, yrange[0]:yrange[1]:self.resolution]
        self.Z = self.data[self.map_offset[0]+self.xrange[0]:self.map_offset[0]+self.xrange[1],
                           self.map_offset[1]+self.yrange[0]:self.map_offset[1]+self.yrange[1]]
        
    def altitude(self, x,y):
        off_x, off_y = self.map_offset
        id_x = int(round(x / self.resolution)) + off_x
        id_y = int(round(y / self.resolution)) + off_y
        if id_x >= self.data.shape[0] or id_y >= self.data.shape[1] or id_x < 0 or id_y < 0:
            return 0
        return self.data[id_x, id_y]

    def max_altitude(self, x, y, radius):
        id_x, id_y = self.pixel_from_coordinate((x,y))
        delta = int(radius / self.resolution)
        return self.data[id_x-delta:id_x+delta, id_y-delta:id_y+delta].max()


    def pixel_from_coordinate(self, point):
        x, y = point
        id_x = int(round(x / self.resolution)) + self.map_offset[0]
        id_y = int(round(y / self.resolution)) + self.map_offset[1]
        return id_x, id_y

    def map_around_position(self, x, y, width, height):
        off_x, off_y = self.map_offset
        x_low = int(round(x / self.resolution)) + off_x - width//2 
        y_low = int(round(y / self.resolution)) + off_y - height//2 
        return self.data[x_low: x_low+width, y_low:y_low+height]

    def get_map(self, p1, p2):
        x1, y1 = self.pixel_from_coordinate(p1)
        x2, y2 = self.pixel_from_coordinate(p2)
        return self.data[x1: x2, y1:y2]

class TerrainClass90m(TerrainClass30m):
    row_length = 6000
    resolution = 90 # in m
    filename = '../deep_glide/SRTM/90m/srtm_38_03.hgt'
    

class TerrainBlockworld(TerrainClass90m):
    
    def __init__(self, ocean=False):
        if ocean:
            self.data = np.zeros((self.row_length,self.row_length))
        else:
            super().__init__()        
        self.blocks = np.zeros((self.row_length,self.row_length))   
        self.map_offset =  [self.row_length//2, self.row_length//2]
        self.create_blocks(10000, False, 100000)
        self.create_blocks(5000, True)

    block_dims = np.array([[30, 10],
                           [60, 10],
                           [90, 10],
                           [10, 30],
                           [10, 60],
                           [10, 90]])
    block_heights = [500., 1000., 2000., 4000.]
    block_spacings = np.array([[10,10]])

    def create_blocks(self, n_blocks, allow_overlap=False, n_tries=0):
        created = 0
        tried = 0
        random.seed()
        n_tries = max(n_blocks, n_tries)
        while created < n_blocks and tried < n_tries:
            tried +=1
            block_spacing = random.choice(self.block_spacings)
            block_dim = random.choice(self.block_dims)
            minx, miny = block_spacing
            maxx, maxy = np.array(self.blocks.shape) - block_spacing - block_dim            
            p1 = np.array([np.random.randint(minx, maxx), np.random.randint(miny, maxy)])
            p2 = p1 + block_dim
            x1, y1 = p1
            x2, y2 = p2
            p1_check = p1 - block_spacing
            p2_check = p2 + block_spacing
            x1c, y1c = p1_check
            x2c, y2c = p2_check
            if allow_overlap or not self.blocks[x1c:x2c, y1c:y2c].any():
                self.blocks[x1:x2, y1:y2].fill(1)
                height = self.data[x1:x2, y1:y2].min() + random.choice(self.block_heights)
                self.data[x1:x2, y1:y2] = np.maximum(self.data[x1:x2, y1:y2], np.full(block_dim, height))
                created +=1


class Config:
    start_ground_distance = (1000,4000)
    goal_ground_distance = (100,100)
    x_range = (-5000, 5000)
    y_range = (-5000, 5000)
    z_range_start = (0, 8000)  
    z_range_goal = (0, 4000)  
    # map_start_range =( (600,3000), (600, 3000)) # for 30m hgt
    map_start_range =( (4200,5400), (2400, 3600)) # for 90m hgt srtm_38_03.hgt
    render_range =( (-15000, 15000), (-15000, 15000)) # for 90m hgt srtm_38_03.hgt
    min_distance_terrain = 50
    ground_distance_radius = 1500


class Env:
    terrain: TerrainClass
    config = Config()

    def __init__(self):
        self.terrain = TerrainBlockworld(False)

    def random_position(self, h_range, radius, z_range):
        rx1,rx2 = self.config.x_range
        ry1,ry2 = self.config.y_range
        rz1,rz2 = z_range
        while True:
            x = np.random.uniform(rx1, rx2)
            y = np.random.uniform(ry1, ry2)
            dmin, dmax = h_range
            dx = np.random.uniform(-1., 1.)
            dy = np.random.uniform(-1., 1.)
            while rx1<=x<=rx2 and ry1<=y<=ry2:
                h = self.terrain.max_altitude(x, y, radius)
                if h+dmin <= rz2 and rz1 <= h+dmax:
                    z = np.random.uniform(max(h + dmin,rz1), min(h+dmax, rz2))
                    return np.array([x,y,z])
                x += dx
                y += dy  
        
    def reset(self) -> object: #->observation
        self.plot_fig = None
        (mx1, mx2), (my1,my2) = self.config.map_start_range
        self.terrain.map_offset = [np.random.randint(mx1, mx2), np.random.randint(my1, my2)]
        self.terrain.define_map_for_plotting(self.config.render_range[0], self.config.render_range[1])              
        self.goal = self.random_position(self.config.goal_ground_distance, self.config.ground_distance_radius, self.config.z_range_goal)
        self.start = self.random_position(self.config.start_ground_distance, self.config.ground_distance_radius, self.config.z_range_start)
        self.goal_orientation = np.zeros(2)
        while np.linalg.norm(self.goal_orientation) == 0: 
            self.goal_orientation = np.random.uniform(-1., 1., 2)
        self.goal_orientation = self.goal_orientation / np.linalg.norm(self.goal_orientation)
        self.pos_offset = self.start.copy()
        self.pos_offset[2] = 0
        self.trajectory=[]   

 
    
    plot_fig: plt.figure = None

    def render(self, mode='human'):
        if self.plot_fig is None:
            self.plot_fig = plt.figure('render 2D', figsize=(10, 10), dpi=80)
            (x1,x2), (y1,y2) = self.config.render_range
            img = self.terrain.get_map((x1,y1), (x2,y2))
            from scipy import ndimage
            img = ndimage.rotate(img, 90)
            plt.clf()
            plt.imshow(img, cmap='gist_earth', vmin=-1000, vmax = 4000, origin='upper', extent=(x1,x2,y1,y2))
            xs,ys, _ = self.start
            xg,yg, _ = self.goal
            xgo, ygo = self.goal_orientation
            plt.plot(xs,ys,'ro')
            plt.plot([xg,xg-xgo*500],[yg,yg-ygo*500], 'b-')
            plt.plot(xg,yg,'ro')
            plt.ion()
            plt.show()            
        plt.figure(self.plot_fig.number)        
        plt.gcf().canvas.draw_idle()
        plt.gcf().canvas.start_event_loop(0.0001)

    flightRenderer3D = None

    def render_episode_3D(self):        
        if self.flightRenderer3D is None:
            from mayavi import mlab
            from pyface.api import GUI
            self.mlab = mlab 
            self.gui = GUI
            self.flightRenderer3D = plotting.Plotting(None, None, None, self.terrain, self.mlab)
            self.gui.process_events()
            self.save_trajectory = True
        self.flightRenderer3D.plot_map(self.terrain)            
        self.flightRenderer3D.plot_start(self.start)
        print('Start Position={} goal Position={}'.format(self.start, self.goal))
        self.flightRenderer3D.plot_goal(self.goal, 500)
        self.flightRenderer3D.plot_path(self.trajectory, radius=10)
        self.gui.process_events()

env = Env()
env.reset()
env.render()
# 3-D-Render kann gerne auch auskommentiert werden. Dann nur 2-D
env.render_episode_3D()
input()