"""
RRT_star 2D
@author: huiming zhou
"""

import math
import mayavi
import numpy as np
from deep_glide.rrt_utils import queue, plotting

from deep_glide.jsbgym_new.sim import SimState, TerrainOcean, SimTimer
from deep_glide.jsbgym_new.sim_handler_rl import SimHandlerRL_SimpleSim
from typing import Dict, List, Tuple
from array import array
from pyface.api import GUI
import matplotlib.pyplot as plt
import time
import signal


def receiveSignal(signalNumber, frame):
    print('Received:', signalNumber)
    rl_trainer.save_model()
    raise SystemExit('Exiting')


class NodeRRT:
    parent = None
    plot = None
    simState: SimState = None
    energy: float


class RL_train:

    terrain: TerrainOcean = TerrainOcean()
    simHandler: SimHandlerRL_SimpleSim

    def save_model(self):
        self.simHandler.save_rl_agents()

    def __init__(self, st_start, x_goal, step_len,
                 goal_sample_rate, search_radius, 
                 iter_max, number_neighbors, number_neighbors_goal,
                 x_range, y_range, z_range, map_offset):
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.number_neighbors = number_neighbors
        self.number_neighbors_goal = number_neighbors_goal
        self.s_start = NodeRRT()
        self.s_start.simState = st_start
        self.terrain.define_map_for_plotting(x_range[1]- x_range[0], y_range[1]- y_range[0], (3601//2 + map_offset[0], 3601//2 + map_offset[1]))
        self.simHandler = SimHandlerRL_SimpleSim(self.s_start.simState, self.terrain, load_models=False)                       
        self.s_goal = NodeRRT()
        self.s_goal.simState = SimState()
        self.s_goal.simState.position = np.array(x_goal)
        self.step_len = step_len
        self.goal_sample_rate = goal_sample_rate
        self.search_radius = search_radius
        self.iter_max = iter_max
        self.vertex = [self.s_start]
        self.path = []
        # ein import von mayavi für das Gesamtprojekt zerstört die JSBSim-Simulation. 
        # Flugrouten werden dann nicht mehr korrekt berechnet - warum auch immer.
        # Deshalb nur lokaler Import.
        from mayavi import mlab
        self.mlab = mlab 
        self.plotting = plotting.Plotting(st_start.position, x_goal, 500, self.terrain, mlab)
        GUI().process_events()

    fig_rewards:plt.figure = None
    plt_rewards = None
    data_rewards = {'x_values': [],
                    'y_values': []}
    n_rewards = 0
            
    rewards = []

    def plot_reward(self, episode, reward):
        xs = self.data_rewards['x_values']
        ys = self.data_rewards['y_values']
        xs.append(episode)
        ys.append(reward)
        if len(ys) < 2: return
        if self.fig_rewards is None:
            self.fig_rewards = plt.figure(0)
            plt.ion()
            plt.show()
        self.fig_rewards = plt.figure(0)
        plt.clf()
        plt.plot(xs, ys)
        plt.gcf().canvas.draw_idle()
        plt.gcf().canvas.start_event_loop(0.3)
        
            

    def guidance(self):        
        self.s_goal.simState.position = np.random.uniform(1000, 2500, 3) * np.random.choice([-1,1],3)
        max_steps = 300
        episode = 0
        for episode in range(0,100):
            self.s_goal.simState.position = np.random.uniform(1000, 2500, 3) * np.random.choice([-1,1],3)
            self.s_goal.simState.position[2] = 0
            self.simHandler.reset_to_state(self.s_start.simState)        
            jsbSimState, arrived, trajectory, rewards = self.simHandler.ttf_maxrange(self.s_goal.simState.position, max_steps, save_trajectory=True)
            self.plotting.plot_goal(self.s_goal.simState.position, 500)
            self.plotting.plot_path(trajectory, radius=10)
            total_reward = np.sum(rewards)
            print('Episode ', episode,': reward min={:.2f} max={:.2f}, mean={:.2f}, med={:.2f} total={:.2f}'.format(np.min(rewards), 
                    np.max(rewards), np.average(rewards), np.median(rewards), total_reward))
            self.plot_reward(episode, total_reward/max_steps)
            GUI().process_events()
        print('Fertig')
        self.simHandler.save_rl_agents()

initial_props={
        'ic/h-sl-ft': 0,#3600./0.3048,
        'ic/long-gc-deg': -2.3273,  # die Koordinaten stimmen nicht mit den Höhendaten überein!
        'ic/lat-geod-deg': 51.3781, # macht aber nix
        'ic/u-fps': 120, #cruise speed Chessna = 120 ft/s
        'ic/v-fps': 0,
        'ic/w-fps': 0,
        'ic/psi-true-rad': 1.0,
    }

class Cursor:
    line = None
    ball = None
    text = None

rl_trainer : RL_train = None
cursor: Cursor = None


def main():
    np.random.seed()
    np.set_printoptions(precision = 2, suppress = True)
    global rl_trainer
    state_start = SimState()
    state_start.props = initial_props
    #state_start.position = np.array([0,0,3500]) # Start Node
    state_start.position = np.array([0,0,3000]) # Start Node
    state_start.props['ic/h-sl-ft'] = state_start.position[2]/0.3048
    
    x_goal = (0, 2000, 0)  # Goal node
    #x_goal = (3180, 1080, 1350+300)  # Goal node
    
    rl_trainer = RL_train(state_start, x_goal, step_len = 3000, goal_sample_rate = 0.10, search_radius = 1000, 
                        iter_max = 500, number_neighbors = 10, number_neighbors_goal = 50,
                        x_range = (-5000, 5000), y_range = (-5000, 5000), z_range = (0, 8000), map_offset=(200,60))
    rl_trainer.guidance()
    rl_trainer.plotting.mlab.show()

if __name__ == '__main__':
    main()
    input()
    exit()
