import jsbsim
import os
from typing import Dict
import time
import numpy as np
from array import array


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

    def define_map(self, xdim, ydim, center_pixel):
        dx = xdim // self.resolution
        dy = ydim // self.resolution
        self.xrange = (-dx*self.resolution/2, dx*self.resolution/2)
        self.yrange = (-dy*self.resolution/2, dy*self.resolution/2)
        X=np.arange(self.xrange[0], self.xrange[1]+self.resolution*2, self.resolution)
        Y=np.arange(self.yrange[0], self.yrange[1]+self.resolution*2, self.resolution)
        self.X, self.Y = np.mgrid[  self.xrange[0]:self.xrange[1]+self.resolution*2:self.resolution, 
                                    self.yrange[0]:self.yrange[1]+self.resolution*2:self.resolution]
        self.Z = self.data[center_pixel[0]-int(dx//2):center_pixel[0]+int(dx//2), center_pixel[1]-int(dy//2):center_pixel[1]+int(dy//2)]
        
    def altitude(self, x,y):
        id_x = int(round((x - self.xrange[0]) / self.resolution))
        id_y = int(round((y - self.yrange[0]) / self.resolution))
        if id_x >= self.Z.shape[0] or id_y >= self.Z.shape[1] or id_x < 0 or id_y < 0:
            return np.math.inf
        return self.Z[id_x, id_y]

class TerrainOcean(TerrainClass):
    def __init__(self):
        self.data = np.zeros((self.row_length,self.row_length)) 

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
