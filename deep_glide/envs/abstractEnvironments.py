from typing import Tuple, List
import numpy as np
from deep_glide.pid import PID, PID_angle
from deep_glide.sim import Sim, SimState, TerrainClass, SimTimer, TerrainOcean
from deep_glide.utils import Normalizer, angle_between
from enum import Enum
import gym
from gym import spaces
import logging
from deep_glide import plotting
from abc import ABC, abstractmethod
from dataclasses import dataclass
from matplotlib import pyplot as plt


class TerminationCondition(Enum):
    NotFinal = 0
    Arrived = 1
    LowerThanTarget = 3
    HitTerrain = 4
    Ground = 5
    OutOfBounds = 6

@dataclass
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
    ground_distance_radius = 500
    initial_props={
        'ic/terrain-elevation-ft': 0.00000001, # 0.0 erzeugt wohl NaNs
        'ic/p-rad_sec': 0,
        'ic/q-rad_sec': 0,
        'ic/r-rad_sec': 0,
        'ic/roc-fpm': 0,
        #'ic/psi-true-rad': 0.0,
        'gear/gear-pos-norm': 0.0, # landing gear raised
        'gear/gear-cmd-norm': 0.0, # lnding gear raised
        'propulsion/set-running': 0, # 1 = running; 0 = off
        'fcs/throttle-cmd-norm': 0.0, # 0.8
        'fcs/mixture-cmd-norm': 0.0, # 0.8
    }

class AbstractJSBSimEnv(gym.Env, ABC):

    config = Config()    
    terrain: TerrainClass = TerrainOcean()

    goal: np.array # = np.array([0.,0.,0.])
    pos: np.array # = np.array([0.,0.,0.])
    start: np.array # = np.array([0.,0.,0.])
    goal_orientation: np.array # = np.array([0.,0.])

    trajectory=[]
    stateNormalizer = Normalizer()

    flightRenderer3D = None
    
    metadata = {'render.modes': ['human']}
    action_space: spaces.Box = None
    observation_space: spaces.Box = None
    sim: Sim
    state: SimState = SimState()

    @abstractmethod
    def _checkFinalConditions(self):
        pass

    @abstractmethod
    def _done(self):
        pass

    @abstractmethod
    def _reward(self):
        pass

    @abstractmethod
    def _get_state(self):
        pass

    def __init__(self, save_trajectory = False):                
        super().__init__()
        self.sim = Sim(sim_dt = 0.02)
        self.save_trajectory = save_trajectory  
        self.m_kg = self.sim.sim['inertia/mass-slugs'] * 14.5939029372
        self.g_fps2 = self.sim.sim['accelerations/gravity-ft_sec2']
        self.initial_state = SimState()
        self.initial_state.props = self.config.initial_props

    def _update(self, sim:Sim ):
        self.pos = sim.pos + self.pos_offset
        self.speed = sim.get_speed_earth()
   
    def step(self, action)->Tuple[object, float, bool, dict]: # ->observation, reward, done, info        
        while True:
            self.sim.run()
            self._update(self.sim)
            if self.timer_pid.check_reset(self.sim.time):
                heading_target = angle_between(np.array([0,1]), action[0:2])
                #pitch_target = vector_pitch(action)
                pitch_target = 0
                roll_target = self.pid_heading(self.sim.sim['attitude/psi-rad'], heading_target)        
                #roll_target=self.pid_heading(self.sim.sim['flight-path/psi-gt-rad'], heading_target)      
                self.sim.sim['fcs/aileron-cmd-norm'] = self.pid_roll(self.sim.sim['attitude/roll-rad'], roll_target)
                #self.sim.sim['fcs/elevator-cmd-norm'] = self.pid_pitch(self.sim.sim['flight-path/gamma-rad'], pitch_target)
                self.sim.sim['fcs/elevator-cmd-norm'] = self.pid_pitch(self.sim.sim['attitude/pitch-rad'], pitch_target)                  
            #if timer_pid_slip.check_reset(self.sim.time):
            #    self.sim.sim['fcs/rudder-cmd-norm'] = self.pid_slip(self.sim.sim['velocities/v-fps'], 0)
            done = self._done()
            if self.timer_goto.check_reset(self.sim.time):# and not goto_arrived:
                if self.save_trajectory: self.trajectory.append(self.pos)
                self.new_state = self._get_state()
                reward = self._reward()
                return self.new_state, reward, done, {}
            if done:
                if self.save_trajectory: self.trajectory.append(self.pos)
                self.new_state = self._get_state()
                reward = self._reward()
                return self.new_state, reward, done, {}

    def random_position(self, h_range, radius, z_range):
        rx1,rx2 = self.config.x_range
        ry1,ry2 = self.config.y_range
        rz1,rz2 = z_range
        while True:
            x = np.random.uniform(rx1, rx2)
            y = np.random.uniform(ry1, ry2)
            dmin, dmax = h_range
            dx, dy  = np.random.uniform(.1, 1.,2)*np.random.choice([-90, 90], 2)
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
        self.goal_orientation = np.random.uniform(.01, 1., 2) * np.random.choice([-1,1],2)
        self.goal_orientation = self.goal_orientation / np.linalg.norm(self.goal_orientation)
        self.pos_offset = self.start.copy()
        self.pos_offset[2] = 0
        self.trajectory=[]   
        self.initial_state.position = self.start
        self._reset_sim_state(self.initial_state)
        #PID-Regler und Timer
        self.pid_pitch = PID_angle('PID pitch', p=-1.5, i=-0.05, d=0, time=0, angle_max=2*np.math.pi, out_min=-1.0, out_max=1.0, anti_windup=1)
        self.pid_roll = PID_angle( 'PID roll', p=17.6, i=0.01, d=35.2, time=0, angle_max=2*np.math.pi, out_min=-1.0, out_max=1.0, anti_windup=1)
        #self.pid_heading = PID_angle('PID heading', p=0.7, i=-0.00002, d=25, time=0, angle_max=2*np.math.pi, out_min=-.6, out_max=.6, anti_windup=1)
        self.pid_heading = PID_angle('PID heading', p=0.7, i=-0.00002, d=25, time=0, angle_max=2*np.math.pi, out_min=-.5*np.math.pi, out_max=.5*np.math.pi, anti_windup=1)
        self.pid_height = PID('PID height', p=0.7, i=-0.00002, d=25, time=0, out_min=-.1, out_max=.1, anti_windup=1)
        self.pid_slip = PID('PID slip', p=0.01, i=0.0, d=0, time=0, out_min=-1.0, out_max=1.0, anti_windup=1) #TODO: pid_slip Tunen
        self.timer_pid = SimTimer(0.04, True)
        self.timer_goto = SimTimer(5.)
        self.timer_pid_slip = SimTimer(0.24, True)
        self.roll_target = 0
        self.pitch_target = 0
        self.heading_target = 0
        self.sim.run()
        self._update(self.sim)
        return self._get_state()
    
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
        x, y, z = self.pos
        z_max  = self.start[2]
        z_min = self.goal[2]        
        color_z = (z-z_min)/(z_max-z_min)
        color_z = 1-max(min(color_z, 1),0)
        plt.plot(x,y, '.', color=(1,color_z,0))
        plt.gcf().canvas.draw_idle()
        plt.gcf().canvas.start_event_loop(0.0001)


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

    plot_episode_2d = None

    def render_episode_2D(self):
        pass
        # if self.plot_episode_2d is None:
        #     self.plot_episode_2d = plt.figure('episode 2D')
        #     img = self.terrain.map_window(self.pos[0], self.pos[1], self.OBS_WIDTH, self.OBS_HEIGHT).copy()            
        #     from scipy import ndimage
        #     img = ndimage.rotate(img, 90)
        #     plt.clf()
        #     plt.imshow(img, cmap='gist_earth', vmin=-1000, vmax = 4000, origin='lower', extent=(0,1000,0,1000))
        #     plt.ion()
        #     plt.show()
        #     self.save_trajectory = True
        # plt.figure(self.plot_episode_2d.number)
        # plt.plot()


    def _reset_sim_state(self, state: SimState, engine_on: bool = False):
        state.position = state.position
        state.props['ic/h-sl-ft'] = state.position[2]/0.3048
        self.sim.reinitialise({**self.config.initial_props, **state.props})
        if engine_on:
            self.sim.start_engines()
            self.sim.set_throttle_mixture_controls(0.8, 0.8)
        #logging.debug("load: ",self.jsbSim.pos[2], state.position[2], state.props['ic/h-sl-ft']*0.3048, self.jsbSim['position/h-sl-ft']*0.3048)
        
    def _get_energy(self):
        speed_fps = np.linalg.norm(self.speed)
        e_kin = 1/2 * self.m_kg * speed_fps**2
        h_ft = self.pos[2]    
        e_pot = self.m_kg * self.g_fps2 * h_ft
        return e_pot+e_kin #in kg*ft^2/sec^2
