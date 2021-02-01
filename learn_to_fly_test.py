import sys
import gym
import numpy as np
import matplotlib.pyplot as plt
from RL_wrapper_gym.DDPG import DDPGagent
from RL_wrapper_gym.utils import *
import torch
import toml
import jsbgym_flex
import jsbgym_flex.properties as prp
from jsbgym_flex.environment import JsbSimEnv
from jsbgym_flex.tasks import *
import random
from rl_experiments import *
from datetime import datetime
from geopy import distance
from runtime_measure import Runtime_Measurement
from rl_visualisation import *
import signal
import os
import time
import sys
import argparse
from live_db import RL_Live_DB
from collections import defaultdict


class RL_Experiment_data:
    cfg: str
    agents = []
    noises = []
    batch_size = 128
    rewards = []
    episode = 0
    save_routes = True
    route_data = defaultdict(list)
    episode_data = defaultdict(list)

    def __init__ (self, torch_device):
        # Initiate the parser
        parser = argparse.ArgumentParser()
        # Add long and short argument
        parser.add_argument("--toml", "-t", help="load configuration from toml file and begin new experiment")
        parser.add_argument("--load_id", "-l", help="load experiment from database and continue experiment")
        parser.add_argument("--enable_fgfs", "-fg", help="enable flightgear output", action="store_true")
        # Read arguments from the command line
        args = parser.parse_args()
        if args.toml:
            print('Loading TOML: {}'.format(args.toml))
            self.cfg = toml.load(args.toml)
            self.experiment = RL_Experiment('./experiments/experiments.db', self.cfg, create = True)
            self.create_env()
            self.create_agents(torch_device)
            
        elif args.load_id:
            print('Loading Experiment with id: {}'.format(args.load_id))
            self.experiment = RL_Experiment('./experiments/experiments.db')
            self.cfg, actor_dicts, critic_dicts, self.rewards, self.episode = self.experiment.load_by_id(args.load_id)   
            self.create_env()
            self.create_agents(torch_device)        
            for agent, actor_dict, critic_dict in zip(self.agents, actor_dicts, critic_dicts) :
                agent.actor.load_state_dict(actor_dict)
                agent.actor_target.load_state_dict(actor_dict)
                agent.critic.load_state_dict(critic_dict)
                agent.critic_target.load_state_dict(critic_dict)
            
        else:
            print('Loading TOML: {}'.format(standard_toml))
            self.cfg = toml.load(standard_toml)
            self.experiment = RL_Experiment('./experiments/experiments.db', self.cfg)
            self.create_env()
            self.create_agents(torch_device)

        if args.enable_fgfs: 
            self.enable_fgfs = True 
        else:
            self.enable_fgfs = False

    def create_env(self):
        self.env = NormalizedEnvMulti(jsbgym_flex.environment.JsbSimEnv(cfg = self.cfg, shaping = Shaping.STANDARD))

    def create_agents(self, device):
        for task in self.env.tasks:
                self.agents.append(DDPGagent(task.observation_space, task.action_space, device, task.actor, task.critic))
                #self.noises.append(OUNoise(task.action_space))
                self.noises.append(OUNoise(task.action_space))
                self.rewards.append([])

    def add_rewards(self, episode_rewards, final_state):
        for r, ep_r in zip(self.rewards, episode_rewards):
            r.append(ep_r)
        experiment.add_rewards(episode_reward, final_state)

    def reset_episode(self):
        self.route_data = defaultdict(list)
        self.episode_data = defaultdict(None)

    def add_routedata(self):
        for prop in self.cfg['environment']['plot_perstep']:
            self.route_data[prop].append(env.get_property(prop))

    def add_episodedata(self):
        for prop in self.cfg['environment']['plot_perepisode']:
            self.episode_data[prop] = env.get_property(prop)
            


def receiveSignal(signalNumber, frame):
    print('Received:', signalNumber)
    experiment.save_state_dicts(data.agents)
    experiment.close()
    raise SystemExit('Exiting')

standard_toml = 'find-target.toml'

if __name__ == "__main__":
    signal.signal(signal.SIGINT, receiveSignal)
    np.set_printoptions(precision=2, suppress=True)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    #device = torch.device("cpu")
    print('Torch Device: {}'.format(device))
        
    data = RL_Experiment_data(device)
    experiment = data.experiment
    env = data.env

    max_steps = data.cfg['gym']['max_steps']
    max_episodes = data.cfg['gym']['max_episodes']
    


    timer = Runtime_Measurement()
    while data.episode < max_episodes or max_episodes <= 0:
        state_n = env.reset()
        for noise in data.noises: noise.reset()
        episode_reward = np.zeros(len(data.agents))
        env.set_property('heading_deg', random.randrange(0,360,1))
        timer.reset()
        data.reset_episode()
        for step in range(max_steps):
            timer.start_sample()
            if step%10 == 0 or data.episode >50: #> 50:
                # Am Anfang actions wiederholen, um 
                # trotz langsamer Reaktion des Flugzeugs zu lernen
                # dann actions bei jedem Schritt
                action_n = [agent.get_action(np.array(state)) for agent, state in zip(data.agents, state_n)]
                action_n = [noise.get_action(action, step) for noise, action in zip(data.noises, action_n)]
                #action_n = [random.uniform(0, 2*math.pi) for agent in data.agents]
            #action_n = -1 + data.episode*0.02
            #action_n = [np.array([action_n])]
            new_state_n, reward_n, done_n, _ = env.step(action_n) 
            for agent, state, action, reward, new_state, done \
                    in zip(data.agents, state_n, action_n, reward_n, new_state_n, done_n):
                agent.memory.push(state, action, reward, new_state, done)
            data.add_routedata()
            for agent in data.agents:
                agent.update(data.batch_size)        
            if (data.episode+1) % 150 == 0 and data.enable_fgfs:
                env.render()
                print('action={} state={} reward={}'.format(action_n, new_state_n, reward_n),end='\r')
            state_n = new_state_n
            episode_reward += reward_n
            timer.stop_sample()
            if done_n[0] or step+1 == max_steps:
                if data.episode % 10 == 0: print()
                sys.stdout.write("episode: {}, reward: {}, time per step={:.3f}ms per episode={:.3f}s \n".
                        format(data.episode, np.round(episode_reward, decimals=2), 
                        timer.average()*1000, timer.total()))
                break
        data.add_episodedata()
        timer.reset()
        timer.start_sample()
        if data.save_routes: 
            experiment.save_route(data.route_data)
            experiment.save_episode(data.episode_data)
        data.add_rewards(episode_reward, state_n)
        timer.stop_sample()
        print(" time plot and safe {:.3f}ms".format(timer.average()*1000))
        data.episode +=1

