from abc import abstractclassmethod, abstractmethod
from deep_glide.utils import elevation_asc2hgt
import jsbsim
import os
from typing import Dict, List
import time
import numpy as np
from array import array
import random


class Sim():

    JSBSIM_DIR = "jsbsim_models"
    AIRCRAFT = 'c172p'
    
    pos = (0,0,0)
    time: int = 0
    sim_dt: int  

    def __init__(self, sim_dt):
        path = os.path.dirname(os.path.realpath(__file__))
        self.sim = jsbsim.FGFDMExec(root_dir=os.path.join(path,self.JSBSIM_DIR))
        self.sim.set_debug_level(0)
        self.sim_dt = sim_dt
        self.initialise(self.sim_dt, self.AIRCRAFT, None)
        self.wall_clock_dt = None
        self.output_enabled = False
        self.sim.disable_output()        

    def __getitem__(self, prop) -> float:
        return self.sim[prop]

    def __setitem__(self, prop, value) -> None:
        self.sim[prop] = value

    def initialise(self, dt: float, model_name: str,
                   init_conditions = None) -> None:
        ic_file = 'minimal_ic.xml'
        ic_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ic_file)
        self.sim.load_ic(ic_path, useStoredPath=False)
        self.sim.load_model(model_name)
        self.sim.set_dt(dt)
        self.set_custom_initial_conditions(init_conditions)
        success = self.sim.run_ic()
        if not success:
            raise RuntimeError('JSBSim failed to init simulation conditions.')


    def set_custom_initial_conditions(self, init_conditions: Dict[str , float] = None) -> None:
        if init_conditions is not None:
            for prop, value in init_conditions.items():
                self.sim[prop] = value

    def reinitialise(self, init_conditions: Dict[str, float] = None) -> None:
        self.set_custom_initial_conditions(init_conditions=init_conditions)
        no_output_reset_mode = 0
        self.sim.reset_to_initial_conditions(no_output_reset_mode)
        self.disable_flightgear_output()
        self.start_lng = self.sim['ic/long-gc-deg']
        self.start_lat = self.sim['ic/lat-geod-deg']        
        self.calcPos()
        self.time = 0

    def run(self) -> bool:
        result = self.sim.run()
        self.time += self.sim_dt
        self.calcPos()
        if self.output_enabled:
            time.sleep(self.wall_clock_dt)
        return result

    def calcPos(self):
        # ENU Koordinatensystem
        # X-Achse: Ost   = positiv
        # Y-Achse: North = positiv
        # Z-Achse: Oben  = positiv
        # Heading: 0° = nach Norden; Drehung im Uhrzeigersinn
        y = self.sim['position/distance-from-start-lat-mt']
        x = self.sim['position/distance-from-start-lon-mt']
        if self.sim['position/lat-geod-deg'] < self.start_lat: y = -y
        if self.sim['position/long-gc-deg'] < self.start_lng: x = -x
        self.pos = np.array([x, y, self.sim['position/h-sl-meters']])
 
    def get_wind(self):
        #ENU
        return np.array( [self.sim['atmosphere/wind-east-fps'],
                          self.sim['atmosphere/wind-north-fps'],
                          -self.sim['atmosphere/wind-down-fps']])

    def set_wind(self, wind: np.array):
        #ENU
        self.sim['atmosphere/wind-east-fps'] = wind[0]
        self.sim['atmosphere/wind-north-fps'] = wind[1]
        self.sim['atmosphere/wind-down-fps'] = -wind[2]

    def get_speed_earth(self):
        #ENU
        return np.array( [self.sim['velocities/v-east-fps'],
                          self.sim['velocities/v-north-fps'],
                          -self.sim['velocities/v-down-fps']])

    def enable_flightgear_output(self):
        if self.wall_clock_dt is not None:
            self.sim.enable_output()
            self.output_enabled = True

    def disable_flightgear_output(self):
        self.sim.disable_output()
        self.output_enabled = False

    def close(self):
        if self.sim:
            self.sim = None

    def set_simulation_time_factor(self, time_factor):
        if time_factor is None:
            self.wall_clock_dt = None
        elif time_factor <= 0:
            raise ValueError('time factor must be positive and non-zero')
        else:
            self.wall_clock_dt = self.sim_dt / time_factor

    def start_engines(self):
        """ Sets all engines running. """
        self['propulsion/set-running'] = -1

    def set_throttle_mixture_controls(self, throttle_cmd: float, mixture_cmd: float):
        """
        Sets throttle and mixture settings
        If an aircraft is multi-engine and has multiple throttle_cmd and mixture_cmd
        controls, sets all of them. Currently only supports up to two throttles/mixtures.
        """
        self['fcs/throttle-cmd-norm'] = throttle_cmd
        self['fcs/mixture-cmd-norm'] = mixture_cmd

        try:
           self['fcs/throttle-cmd-norm[1]'] = throttle_cmd
           self['fcs/mixture-cmd-norm[1]'] = mixture_cmd
        except KeyError:
            pass  # must be single-control aircraft

    def set_landing_gear(self, value):
        """ Raises all aircraft landing gear. """
        self['gear/gear-pos-norm'] = value
        self['gear/gear-cmd-norm'] = value


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
    filename = 'SRTM/90m/srtm_38_03.hgt'

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

    block_dimensions = np.array([[30, 10],
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
            block_dim = random.choice(self.block_dimensions)
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

class TerrainOcean(TerrainClass90m):
    def __init__(self):
        self.data = np.zeros((self.row_length,self.row_length))
        self.map_offset =  [self.row_length//2, self.row_length//2]

    def altitude(self, x, y):
        return 0.
 


class SimTimer:
    def __init__(self, interval_s: float, fire_on_start = False):
        self._interval = interval_s        
        if fire_on_start:
            self._last_time = -interval_s   
        else:
            self._last_time = 0    

    def reset(self, jsbSim_time_s: float):
        self._last_time = jsbSim_time_s

    def check_reset(self, jsbSim_time_s: float) -> bool:
        if jsbSim_time_s >= self._last_time + self._interval:
            self._last_time = jsbSim_time_s
            return True
        else: 
            return False

    def check(self, jsbSim_time_s: float) -> bool:
        if jsbSim_time_s > self._last_time + self._interval:
            return True
        else: 
            return False

class SimState:
    props: Dict[str , float] = {}
    position: np.array = None
