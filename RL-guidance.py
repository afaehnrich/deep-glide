"""
RRT_star 2D
@author: huiming zhou
"""

import math
import numpy as np
#from rrt_utils import queue, plotting

from deep_glide.pid import PID_angle, PID
from deep_glide.sim import Sim,  SimState, TerrainClass, TerrainOcean, SimTimer, TerrainClass90m, TerrainClass30m, TerrainBlockworld
from deep_glide.envs.withoutMap import JSBSimEnv_v0, JSBSimEnv_v1, JSBSimEnv_v2, JSBSimEnv_v4, JSBSimEnv_v5
from deep_glide.envs.withMap import JSBSimEnv2D_v0
from typing import Dict, List, Tuple
from array import array
import matplotlib.pyplot as plt
import time
import signal
import torch
from deep_glide.deprecated.rl_wrapper.RL_wrapper import DDPGagent
from deep_glide.deprecated.rl_wrapper.utils import OUNoise, NormalizedEnv
from deep_glide.deprecated.rl_wrapper.model import actors, critics
import logging
from datetime import datetime
#from pyface.api import GUI

class RL_train:

    simHandler =  JSBSimEnv_v0() #JSBSimEnv2D_v0()
    BATCH_SIZE = 128
 
    def init_rl_agents(self, action_space, obs_space, load_models):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print('torch device: CUDA')
        else:
            self.device = torch.device("cpu")
            print('torch device: cpu')
        self.agent = DDPGagent(action_space, obs_space, device=self.device, actor_type=actors.lin_4x128, 
                                critic_type=critics.lin_4x128, load_from_disk=load_models)
        self.noise = OUNoise(action_space)
        self.normalizer = NormalizedEnv(action_space)

    def save_model(self):
        self.agent.save_model()

    def __init__(self, st_start, x_goal, step_len,
                 goal_sample_rate, search_radius, 
                 iter_max, number_neighbors, number_neighbors_goal):
        self.st_start = st_start
        self.simHandler.terrain = TerrainBlockworld()
        # ein import von mayavi für das Gesamtprojekt zerstört die JSBSim-Simulation. 
        # Flugrouten werden dann nicht mehr korrekt berechnet - warum auch immer.
        # Deshalb nur lokaler Import.
        # from mayavi import mlab
        # self.mlab = mlab 
        # self.flightRenderer = plotting.Plotting(None, None, None, self.simHandler.terrain, self.mlab)
        # GUI().process_events()


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
        
            
    def guidance(self, render=True, max_steps = 300, max_episodes = 100):
        self.init_rl_agents(self.simHandler.action_space, self.simHandler.observation_space, load_models=False)
        logging.basicConfig(level=logging.INFO) 
        for episode in range(0,max_episodes):
            rewards = []
            state = self.simHandler.reset()
            time1 = datetime.now()
            for step in range(0, max_steps):
                # ->observation, reward, done, info  
                action=np.random.uniform(-1.,1.,2)
                action = self.agent.get_action(state)
                action = self.noise.get_action(action, step)
                action = self.normalizer.action(action)
                new_state, reward, done, _ = self.simHandler.step(action)
                rewards.append(reward)            
                self.agent.memory.push(state, action, reward, new_state, done)
                self.agent.update(self.BATCH_SIZE)        
                state = new_state                
                if done: break            
            time2 = datetime.now()
            if render: self.simHandler.render()
            total_reward = np.sum(rewards)
            print('Episode ', episode,': reward min={:.2f} max={:.2f}, mean={:.2f}, med={:.2f} total={:.2f}   time={:.1f}s'.format(np.min(rewards), 
                    np.max(rewards), np.average(rewards), np.median(rewards), total_reward, (time2-time1).total_seconds()))
            self.plot_reward(episode, total_reward/max_steps)
            #GUI().process_events()
        print('Fertig')
        self.simHandler.save_rl_agents()

    def guidance_perfect(self, render=True, max_steps = 300, max_episodes = 100):
        logging.basicConfig(level=logging.INFO) 
        for episode in range(0,max_episodes):
            rewards = []
            state = self.simHandler.reset()
            time1 = datetime.now()
            for step in range(0, max_steps):
                pos = self.simHandler.pos[0:2]
                goal = self.simHandler.goal[0:2]
                dir = goal-pos
                # ->observation, reward, done, info  
                dir_len =  np.linalg.norm(dir)
                if dir_len == 0: dir_len = 1.
                action=dir/ np.linalg.norm(dir)
                new_state, reward, done, _ = self.simHandler.step(action)
                state = new_state
                rewards.append(reward)            
                if done: break            
            time2 = datetime.now()
            if render: self.simHandler.render()
            total_reward = np.sum(rewards)
            print('Episode ', episode,': reward min={:.2f} max={:.2f}, mean={:.2f}, med={:.2f} total={:.2f}  episode_len={} time={:.1f}s'.format(np.min(rewards), 
                    np.max(rewards), np.average(rewards), np.median(rewards), total_reward, step, (time2-time1).total_seconds()))                    
            self.plot_reward(episode, total_reward/max_steps)
            # print('Start: {} Goal: {}'.format(self.simHandler.start, self.simHandler.goal))
            # input()
            #GUI().process_events()
        print('Fertig')
        self.simHandler.save_rl_agents()

    def height(self, max_steps = 300, max_episodes = 1000):
        logging.basicConfig(level=logging.INFO) 
        energy = []
        distance = []
        for episode in range(0,max_episodes):
            state = self.simHandler.reset()
            energy.append(self.simHandler._get_energy())
            distance.append(np.linalg.norm(self.simHandler.goal[0:2]-self.simHandler.pos[0:2]))
            print(episode,end='\r')
        x = energy
        print('Energy min={:.2f} max={:.2f}, mean={:.2f}, med={:.2f} '.format(np.min(x), 
                    np.max(x), np.average(x), np.median(x)))
        x = distance
        print('Distance min={:.2f} max={:.2f}, mean={:.2f}, med={:.2f} '.format(np.min(x), 
                    np.max(x), np.average(x), np.median(x)))
        print('Fertig')

    def guidance_random(self, render=True, max_steps = 300, max_episodes = 25):
        logging.basicConfig(level=logging.INFO) 
        not_final_reward=[]
        for episode in range(0,max_episodes):
            rewards = []
            state = self.simHandler.reset()
            time1 = datetime.now()
            for step in range(0, max_steps):
                pos = np.array(state[5:8])
                goal = np.array(state[8:11])
                dir = goal-pos
                # ->observation, reward, done, info  
                dir_len =  np.linalg.norm(dir)
                if dir_len == 0: dir_len = 1.
                action=self.simHandler.action_space.sample()
                new_state, reward, done, _ = self.simHandler.step(action)
                if not done: not_final_reward.append(reward)
                state = new_state
                rewards.append(reward)            
                if done: break            
                if render: self.simHandler.render()
            time2 = datetime.now()
            #if render: 
            #    self.simHandler.render_episode_3D()
            total_reward = np.sum(rewards)
            print('Episode ', episode,': reward min={:.2f} max={:.2f}, mean={:.2f}, med={:.2f} total={:.2f}  episode_len={} time={:.1f}s'.format(np.min(rewards), 
                    np.max(rewards), np.average(rewards), np.median(rewards), total_reward, step, (time2-time1).total_seconds()))
            self.plot_reward(episode, total_reward/max_steps)
            input()
            #GUI().process_events()
        x = not_final_reward
        print('Reward not final min={:.5f} max={:.5f}, mean={:.5f}, med={:.5f} total per episode={:.5f}'.format(np.min(x), 
                    np.max(x), np.average(x), np.median(x), np.sum(x)/max_episodes))
        print('Fertig')

    def show_map(self):
        logging.basicConfig(level=logging.INFO) 
        state = self.simHandler.reset()
        self.simHandler.render()
        input()
        

initial_props={
        'ic/h-sl-ft': 0,#3600./0.3048,
        'ic/long-gc-deg': -2.3273,  # die Koordinaten stimmen nicht mit den Höhendaten überein!
        'ic/lat-geod-deg': 51.3781, # macht aber nix
        'ic/u-fps': 120, #cruise speed Chessna = 120 ft/s
        'ic/v-fps': 0,
        'ic/w-fps': 0,
        'ic/psi-true-rad': 0.0,
    }
rl_trainer : RL_train = None

def main():
    np.random.seed()
    global rl_trainer
    state_start = SimState()
    state_start.props = initial_props
    #state_start.position = np.array([0,0,3500]) # Start Node
    state_start.position = np.array([0,0,3000]) # Start Node
    state_start.props['ic/h-sl-ft'] = state_start.position[2]/0.3048
    
    x_goal = (0, 2000, 0)  # Goal node
    #x_goal = (3180, 1080, 1350+300)  # Goal node
    
    rl_trainer = RL_train(state_start, x_goal, step_len = 3000, goal_sample_rate = 0.10, search_radius = 1000, 
                        iter_max = 500, number_neighbors = 10, number_neighbors_goal = 50)
    # rl_trainer.guidance_perfect() # Funktioniert nicht mit normalisierten States!
    # rl_trainer.height()
    rl_trainer.guidance_random( render=True)
    # rl_trainer.show_map()
    # rl_trainer.guidance()

if __name__ == '__main__':    
    main()
    input()
    exit()
