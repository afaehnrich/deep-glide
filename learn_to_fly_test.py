import sys
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from RL_wrapper_gym.DDPG import DDPGagent
from RL_wrapper_gym.utils import *
import torch
import toml
import gym_jsbsim_simple
import gym_jsbsim_simple.properties as prp
from gym_jsbsim_simple.environment import JsbSimEnv
from gym_jsbsim_simple.tasks import *
import random

np.set_printoptions(precision=2, suppress=True)
enable_fgfs = False

cfg = toml.load('gym-jsbsim-cfg.toml')

env = NormalizedEnv(gym_jsbsim_simple.environment.JsbSimEnv(cfg = cfg, 
        task_type = AFHeadingControlTask, shaping = Shaping.STANDARD))
#env = NormalizedEnv(gym_jsbsim_simple.environment.JsbSimEnv(cfg = cfg, 
#        task_type = FlyAlongLineTask, shaping = Shaping.STANDARD))
#device = torch.device("cpu")
device = torch.device("cuda")
agent = DDPGagent(env, device)
noise = OUNoise(env.action_space)
batch_size = 128
rewards = []
avg_rewards = []
p1 = plt.subplot(2,1,1)
p2 = plt.subplot(2,1,2)
p1.plot(0,0,'.')
plt.ion()
plt.show()   
plt.pause(0.001)        
for episode in range(0,150,1):
    state = env.reset()
    noise.reset()
    episode_reward = 0
    env.set_property('heading_deg', random.randrange(0,360,1))
    routex=[]
    routey=[]
    for step in range(500):
        if step%10 == 0 or episode > 50:
            # Am Anfang actions wiederholen, um 
            # trotz langsamer Reaktion des Flugzeugs zu lernen
            # dann actions bei jedem Schritt
            action = agent.get_action(state)
            action = noise.get_action(action, step)
        new_state, reward, done, _ = env.step(action) 
        agent.memory.push(state, action, reward, new_state, done)
        routex.append(env.get_property('lat_geod_deg'))
        routey.append(env.get_property('lng_geoc_deg'))
        if len(agent.memory) > batch_size:
            agent.update(batch_size)        
        if (episode+1) % 150 == 0 and enable_fgfs:
            env.render()
            print('action={} state={} reward={}'.format(action, new_state, reward),end='\r')
        state = new_state
        episode_reward += reward
        if done:
            if episode % 10 == 0: print()
            sys.stdout.write("episode: {}, reward: {}, average _reward: {:.2f} \n".
                    format(episode, np.round(episode_reward, decimals=2), 
                    np.mean(rewards[-10:])))
            p1.clear()
            p1.plot(routex,routey,'.')
            plt.show()
            plt.pause(0.001)        
            break
    rewards.append(episode_reward)
    avg_rewards.append(np.mean(rewards[-10:]))
    p2.clear()
    p2.plot(rewards)
    p2.plot(avg_rewards)

plt.ioff()
plt.show()
