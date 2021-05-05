from typing import Tuple, List
import numpy as np
from deep_glide.pid import PID, PID_angle
from deep_glide.sim import Sim, SimState, TerrainClass, SimTimer, TerrainOcean, Runway
from deep_glide.utils import Normalizer, angle_between, ensure_dir, ensure_newfile
from enum import Enum
import gym
from gym import spaces
import logging
from deep_glide import plotting
from abc import ABC, abstractmethod
from dataclasses import dataclass
from matplotlib import pyplot as plt
import math
import os
from datetime import date
from deep_glide.envs.abstractEnvironments import AbstractJSBSimEnv, Config

class AbstractJSBSimRawEnv(AbstractJSBSimEnv):

    def step(self, action:np.ndarray)->Tuple[object, float, bool, dict]: # ->observation, reward, done, info        
        action = action.clip(-1., 1.)
        self.sim.sim['fcs/aileron-cmd-norm'] = action[0]
        self.sim.sim['fcs/elevator-cmd-norm'] = action[1]
        self.sim.sim['fcs/rudder-cmd-norm'] = action[2]
        while True:
            self.sim.run()
            self._update(self.sim)
            done = self._done()
            if self.timer_action.check_reset(self.sim.time):# and not goto_arrived:
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
        np.random.seed()
        if self.episode_rendered: 
            self.save_plot()
            self.episode_rendered = False
        (mx1, mx2), (my1,my2) = self.config.map_start_range
        self.terrain.map_offset = [np.random.randint(mx1, mx2), np.random.randint(my1, my2)]
        self.terrain.define_map_for_plotting(self.config.render_range[0], self.config.render_range[1])              
        self.goal = self.random_position(self.config.goal_ground_distance, self.config.ground_distance_radius, 
                                        self.config.x_range_goal, self.config.y_range_goal, self.config.z_range_goal)
        self.start = self.random_position(self.config.start_ground_distance, self.config.ground_distance_radius,
                                          self.config.x_range_start, self.config.y_range_start, self.config.z_range_start)
        self.goal_orientation = np.random.uniform(.01, 1., 3) * np.random.choice([-1,1],3)
        self.goal_orientation[2] = 0
        self.goal_orientation = self.goal_orientation / np.linalg.norm(self.goal_orientation)
        self.runway = Runway(self.goal[0:2], self.goal_orientation[0:2],self.config.runway_dimension)
        self.terrain.set_runway(self.runway)
        self.pos_offset = self.start.copy()
        self.pos_offset[2] = 0
        self.trajectory=[]   
        self.initial_state.position = self.start
        self._reset_sim_state(self.initial_state)
        self.timer_action = SimTimer(.04)
        self.sim.run()
        self._update(self.sim)
        self.episode +=1
        return self._get_state()
