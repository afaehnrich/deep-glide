import jsbsim
import os
from typing import Dict, Tuple, List
import time
import numpy as np
from deep_glide.jsbgym_new.pid import PID, PID_angle
from deep_glide.jsbgym_new.guidance import TrackToFix3D, angle_between, vector_pitch
from deep_glide.jsbgym_new.sim import Sim, SimState, TerrainClass, SimTimer, TerrainOcean
from deep_glide.jsbgym_new.guidance import TrackToFix, TrackToFixHeight
from deep_glide.jsbgym_new.abstractSimHandler import AbstractJSBSimEnv, TerminationCondition
from array import array
from deep_glide.jsbgym_new.properties import BoundedProperty, Properties, PropertylistToBox
from enum import Enum
import gym
from gym import spaces
import logging
from deep_glide.rrt_utils import plotting
from abc import ABC, abstractmethod

class AbstractSimHandlerGoal(gym.GoalEnv, ABC):
    

    action_props: List[BoundedProperty] = None
    observation_props: List[BoundedProperty] = None
    flightRenderer3D = None
    terrain: TerrainClass = None
    goal: Tuple[float] = None

    metadata = {'render.modes': ['human']}
    action_space: spaces.Box = None
    observation_space: spaces.Dict
    sim = Sim(sim_dt = 0.02)
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
    def _get_state(self):
        pass

    @abstractmethod
    def _reward(self, terminal_condition) -> float:
        pass

    def __init__(self, initial_State: SimState, save_trajectory = False):        
        super(AbstractJSBSimEnv, self).__init__()
        x_range = (-5000, 5000)
        y_range = (-5000, 5000)
        z_range = (0, 8000)
        self.save_trajectory = save_trajectory
        map_offset=(200,60)      
        self.terrain.define_map(x_range[1]- x_range[0], y_range[1]- y_range[0], (3601//2 + map_offset[0], 3601//2 + map_offset[1]))
        self._reset_sim_state(initial_State)
        self.m_kg = self.sim.sim['inertia/mass-slugs'] * 14.5939029372
        self.g_fps2 = self.sim.sim['accelerations/gravity-ft_sec2']
        self.initial_state = initial_State
        self.start = np.array(initial_State.position)
        self.action_space = spaces.Box(*PropertylistToBox(self.action_props))
        self.observation_space = self.observation_space = spaces.Dict({
                'observation': spaces.Box(*PropertylistToBox(self.observation_props)),
                'achieved_goal': spaces.Discrete(2 ** n_bits - 1),
                'desired_goal': spaces.Discrete(2 ** n_bits - 1)
            })
        self.obs_space = spaces.Box(*PropertylistToBox(self.observation_props))

    
    def step(self, action)->Tuple[object, float, bool, dict]: # ->observation, reward, done, info        
        while True:
            self.sim.run()
            if self.save_trajectory: self.trajectory.append(self.sim.pos)
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
            terminalCondition = self._checkFinalConditions()
            if terminalCondition == TerminationCondition.NotFinal:
                 done=False
            else: done=True
            if self.timer_goto.check_reset(self.sim.time):# and not goto_arrived:
                self.new_state = self._get_state()
                reward = self._reward(terminalCondition)
                return self.new_state, reward, done, {}
            if done:     
                self.new_state = self._get_state()
                reward = self._reward(terminalCondition)
                return self.new_state, reward, done, {}

    def reset(self) -> object: #->observation
        self._reset_sim_state(self.initial_state)
        
        self.goal = np.random.uniform(3000, 5000, 3) * np.random.choice([-1,1],3)
        self.goal[2] = self.terrain.altitude(self.goal[0], self.goal[1])+ 100

        #PIDs und Timer
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
        self.position_0 = np.array(self.sim.pos)
        if self.save_trajectory:
            self.trajectory=[] 
        else:
            self.trajectory = None
        self.sim.run()
        return self._get_state()

    def render(self, mode='human'):        
        if self.flightRenderer3D is None:
             from mayavi import mlab
             from pyface.api import GUI
             self.mlab = mlab 
             self.gui = GUI
             self.flightRenderer3D = plotting.Plotting(None, None, None, self.terrain, self.mlab)
             self.gui.process_events()
        self.flightRenderer3D.plot_start(self.start)
        self.flightRenderer3D.plot_goal(self.goal, 500)
        self.flightRenderer3D.plot_path(self.trajectory, radius=10)
        self.gui.process_events()

    def _reset_sim_state(self, state: SimState, engine_on: bool = False):
        self.state = SimState()
        self.state.props = state.props.copy()
        self.state.position = state.position
        self.sim.reinitialise({**self.initial_props, **state.props}, state.position[0:2])
        if engine_on:
            self.sim.start_engines()
            self.sim.set_throttle_mixture_controls(0.8, 0.8)
        #logging.debug("load: ",self.jsbSim.pos[2], state.position[2], state.props['ic/h-sl-ft']*0.3048, self.jsbSim['position/h-sl-ft']*0.3048)

    def _checkFinalConditions(self):
        if np.linalg.norm(self.goal[0:2] - self.sim.pos[0:2])<500:
            logging.debug('Arrived at Target')
            return TerminationCondition.Arrived
        elif self.sim.pos[2]<self.goal[2]-10:
            logging.debug('   Too low: ',self.sim.pos[2],' < ',self.goal[2]-10)
            return TerminationCondition.LowerThanTarget
        elif self.sim.pos[2]<=self.terrain.altitude(self.sim.pos[0], self.sim.pos[1])+ self.min_distance_terrain:
            logging.debug('   Terrain: {:.1f} <= {:.1f}+{:.1f}'.format(self.sim.pos[2],
                    self.terrain.altitude(self.sim.pos[0], self.sim.pos[1]), self.min_distance_terrain))
            return TerminationCondition.HitTerrain
        else: return TerminationCondition.NotFinal
          
    def _get_energy(self):
        speed_fps = np.linalg.norm(self.sim.get_speed_earth())
        e_kin = 1/2 * self.m_kg * speed_fps**2
        h_ft = self.sim['position/h-sl-ft']    
        e_pot = self.m_kg * self.g_fps2 * h_ft
        return e_pot+e_kin #in kg*ft^2/sec^2


class SimHandlerGoal(AbstractSimHandlerGoal): 
    metadata = {'render.modes': ['human']}

    terrain: TerrainClass = TerrainOcean()
    goal: Tuple[float] = (0.,0.,0.)

    action_props = [Properties.custom_dir_x,
                    Properties.custom_dir_y]
    #               ,rl_wrapper.properties.Properties.custom_dir_z]

    observation_props = [Properties.attitude_psi_rad,
                        Properties.attitude_roll_rad,
                        Properties.velocities_p_rad_sec,
                        Properties.velocities_q_rad_sec,
                        Properties.velocities_r_rad_sec,
                        Properties.infinity,
                        Properties.infinity,
                        Properties.infinity,
                        Properties.infinity,
                        Properties.infinity,
                        Properties.infinity,
                        Properties.infinity,
                        Properties.infinity,
                        Properties.infinity]

    def __init__(self):
        initial_props={
                'ic/h-sl-ft': 0,#3600./0.3048,
                'ic/long-gc-deg': -2.3273,  # die Koordinaten stimmen nicht mit den Höhendaten überein!
                'ic/lat-geod-deg': 51.3781, # macht aber nix
                'ic/u-fps': 120, #cruise speed Chessna = 120 ft/s
                'ic/v-fps': 0,
                'ic/w-fps': 0,
                'ic/psi-true-rad': 1.0,
            }
        state_start = SimState()
        state_start.props = initial_props
        #state_start.position = np.array([0,0,3500]) # Start Node
        state_start.position = np.array([0, 0, 3000])  #  Start Node
        state_start.props['ic/h-sl-ft'] = state_start.position[2]/0.3048
        return super().__init__(state_start, save_trajectory=False)

    def _get_state(self):
        speed = self.sim.get_speed_earth()
        return np.array([self.sim.sim['attitude/psi-rad'],
                        self.sim.sim['attitude/roll-rad'],
                        self.sim.sim['velocities/p-rad_sec'],
                        self.sim.sim['velocities/q-rad_sec'],
                        self.sim.sim['velocities/r-rad_sec'],
                        self.sim.pos[0],
                        self.sim.pos[1],
                        self.sim.pos[2],
                        self.goal[0],
                        self.goal[1],
                        self.goal[2],
                        speed[0],
                        speed[1],
                        speed[2]
                        ])

    def _reward(self, terminal_condition):
        if terminal_condition == TerminationCondition.NotFinal:
            dir_target = self.goal-self.sim.pos
            v_aircraft = self.sim.get_speed_earth()            
            angle = angle_between(dir_target[0:2], v_aircraft[0:2])
            if angle == 0: return 0.
            return -abs(angle_between(dir_target[0:2], v_aircraft[0:2]))/np.math.pi / 100.       
        if terminal_condition == TerminationCondition.Arrived: return +10.
        dist_target = np.linalg.norm(self.goal[0:2]-self.sim.pos[0:2])
        return -dist_target/3000.

    # def _reward(self, terminal_condition):
    #     dist_target = np.linalg.norm(self.goal[0:2]-self.sim.pos[0:2])
    #     dist_to_ground = self.sim.pos[2] - self.terrain.altitude(self.sim.pos[0], self.sim.pos[1])
    #     #dist_max = np.linalg.norm(np.array([0,0]) - np.array(self.goal[0:2]))
    #     #reward = -np.log(dist_target/dist_to_ground)
    #     reward = -np.log(dist_target/800)-dist_target/dist_to_ground
    #     if terminal_condition == TerminationCondition.Arrived: reward +=15000.
    #     if terminal_condition == TerminationCondition.Ground: reward -= 5000.
    #     elif terminal_condition == TerminationCondition.HitTerrain: reward -= 5000.
    #     elif terminal_condition == TerminationCondition.LowerThanTarget: reward -= 5000.        
    #     return reward


    sim = Sim(sim_dt = 0.02)
    state: SimState = SimState()
    terrain: TerrainClass
    min_distance_terrain = 100
    terrain: TerrainClass = TerrainOcean() 

    goal = (0,0,0)

    # def reward_head(self, step, max_steps):
    #     dir_target = self.goal-self.sim.pos
    #     #dir_target = self.goal-self.start
    #     v_aircraft = self.sim.get_speed_earth()
    #     reward = -abs(angle_between(dir_target[0:2], v_aircraft[0:2]))/np.math.pi
    #     done = False
    #     return self.reward_check_final(reward, done, step, max_steps)

    # def reward_distance(self, step, max_steps):
    #     dist_target = np.linalg.norm(self.goal[0:2]-self.sim.pos[0:2])
    #     dist_max = np.linalg.norm(np.array([0,0]) - np.array(self.goal[0:2]))
    #     reward = -np.clip(-1., 1., dist_target/dist_max)
    #     done = False
    #     return self.reward_check_final(reward, done, step, max_steps)


    # def reward_head_original(self, terminal_condition):
    #     dir_target = self.goal-self.sim.pos
    #     #dir_target = self.goal-self.start
    #     v_aircraft = self.sim.get_speed_earth()
    #     reward = -abs(angle_between(dir_target[0:2], v_aircraft[0:2]))
    #     if terminal_condition == TerminationCondition.Arrived: reward +=50
    #     elif terminal_condition == TerminationCondition.Ground: reward = reward *100
    #     elif terminal_condition == TerminationCondition.HitTerrain: reward = reward *100
    #     elif terminal_condition == TerminationCondition.LowerThanTarget: reward = reward * 100
    #     return reward
