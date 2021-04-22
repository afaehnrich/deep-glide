from typing import Tuple, List
import numpy as np
from deep_glide.jsbgym_new.pid import PID, PID_angle
from deep_glide.jsbgym_new.guidance import angle_between
from deep_glide.jsbgym_new.sim import Sim, SimState, TerrainClass, SimTimer, TerrainOcean
from deep_glide.jsbgym_new.utils import Normalizer
from deep_glide.jsbgym_new.properties import BoundedProperty, PropertylistToBox
from enum import Enum
import gym
from gym import spaces
import logging
from deep_glide.rrt_utils import plotting
from abc import ABC, abstractmethod

class TerminationCondition(Enum):
    NotFinal = 0
    Arrived = 1
    LowerThanTarget = 3
    HitTerrain = 4
    Ground = 5
    OutOfBounds = 6

class AbstractJSBSimEnv(gym.Env, ABC):
    
    terrain: TerrainClass = TerrainOcean()
    goal = np.array([0.,0.,0.])
    pos = np.array([0.,0.,0.])
    start = np.array([0.,0.,0.])
    min_distance_terrain = 100
    trajectory=[]
    stateNormalizer = Normalizer()

    action_props: List[BoundedProperty] = None
    observation_props: List[BoundedProperty] = None
    flightRenderer3D = None
    
    metadata = {'render.modes': ['human']}
    action_space: spaces.Box = None
    observation_space: spaces.Box
    sim: Sim
    state: SimState = SimState()
    min_distance_terrain = 100
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

    def __init__(self, initial_State: SimState, save_trajectory = False):                
        super().__init__()      
        self.sim = Sim(sim_dt = 0.02)
        self.x_range = (-5000, 5000)
        self.y_range = (-5000, 5000)
        self.z_range = (2000, 4000)
        self.z_range_goal = (100, 101)
        self.save_trajectory = save_trajectory
        map_offset=(200,60)      
        self.terrain.define_map_for_plotting(self.x_range[1]- self.x_range[0], self.y_range[1]- self.y_range[0], (3601//2 + map_offset[0], 3601//2 + map_offset[1]))
        self._reset_sim_state(initial_State)
        self.m_kg = self.sim.sim['inertia/mass-slugs'] * 14.5939029372
        self.g_fps2 = self.sim.sim['accelerations/gravity-ft_sec2']
        self.initial_state = initial_State
        self.start = np.array(initial_State.position)
        self.action_space = spaces.Box(*PropertylistToBox(self.action_props))
        self.observation_space = spaces.Box(*PropertylistToBox(self.observation_props))

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

    def reset(self) -> object: #->observation
        start_height = np.random.uniform(0, self.z_range[1]- self.z_range[0]) + self.z_range[0] \
                                         + self.terrain.altitude(self.start[0], self.start[1])  
        self.start[0] = np.random.uniform(0, self.x_range[1]- self.x_range[0]) + self.x_range[0]
        self.start[1] = np.random.uniform(0, self.y_range[1]- self.y_range[0]) + self.y_range[0]
        self.start[2] = start_height
        self.goal[0] = np.random.uniform(0, self.x_range[1]- self.x_range[0]) + self.x_range[0]
        self.goal[1] = np.random.uniform(0, self.y_range[1]- self.y_range[0]) + self.y_range[0]
        self.goal[2] = np.random.uniform(0, self.z_range_goal[1]- self.z_range_goal[0]) + self.z_range_goal[0] \
                                        + self.terrain.altitude(self.goal[0], self.goal[1])   
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

    def render(self, mode='human'):        
        if self.flightRenderer3D is None:
             from mayavi import mlab
             from pyface.api import GUI
             self.mlab = mlab 
             self.gui = GUI
             self.flightRenderer3D = plotting.Plotting(None, None, None, self.terrain, self.mlab)
             self.gui.process_events()
             self.save_trajectory = True
        self.flightRenderer3D.plot_start(self.start)
        print('Start Position={} goal Position={}'.format(self.start, self.goal))
        self.flightRenderer3D.plot_goal(self.goal, 500)
        self.flightRenderer3D.plot_path(self.trajectory, radius=10)
        self.gui.process_events()

    def _reset_sim_state(self, state: SimState, engine_on: bool = False):
        state.position = state.position
        state.props['ic/h-sl-ft'] = state.position[2]/0.3048
        self.sim.reinitialise({**self.initial_props, **state.props})
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
