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

np.set_printoptions(precision=2, suppress=True)
enable_fgfs = False
max__travel_dist = 0.15

cfg = toml.load('heading-control.toml')
#cfg = toml.load('fly-along-line.toml')

env = NormalizedEnvMulti(jsbgym_flex.environment.JsbSimEnv(cfg = cfg, shaping = Shaping.STANDARD))
#env = NormalizedEnv(jsbgym_flex.environment.JsbSimEnv(cfg = cfg, 
#        task_type = FlyAlongLineTask, shaping = Shaping.STANDARD))
if torch.cuda.is_available():
     device = torch.device("cuda")
else:
    device = torch.device("cpu")
print('Torch Device: {}'.format(device))
agents = []
noises = []
for task in env.tasks:
    agents.append(DDPGagent(task.observation_space, task.action_space, device, task.actor, task.critic))
    noises.append(OUNoise(task.action_space))
batch_size = 128
rewards = []
avg_rewards = []
plt.figure()
#p1 = plt.subplot(2,1,1)
#p2 = plt.subplot(2,1,2)
latstart = cfg.get('environment').get('initial_state').get('initial_latitude_geod_deg')
lonstart = cfg.get('environment').get('initial_state').get('initial_longitude_geoc_deg')
plt.plot(0,0,'.')
plt.figure()
plt.ion()
plt.show()   
plt.pause(0.001)        
t_head=0
for episode in range(0,150,1):
    state_n = env.reset()
    for noise in noises: noise.reset()
    episode_reward = np.array([0.])
    env.set_property('heading_deg', random.randrange(0,360,1))
    routex=[]
    routey=[]
    #env.set_property('target_heading', t_head)
    for step in range(500):
        if step%10 == 0 or episode > 50:
            # Am Anfang actions wiederholen, um 
            # trotz langsamer Reaktion des Flugzeugs zu lernen
            # dann actions bei jedem Schritt
            action_n = [agent.get_action(state) for agent, state in zip(agents, state_n)]
            action_n = [noise.get_action(action, step) for action in action_n]
        new_state_n, reward_n, done_n, _ = env.step(action_n) 
        for agent, state, action, reward, new_state, done \
                in zip(agents, state_n, action_n, reward_n, new_state_n, done_n):
            agent.memory.push(state, action, reward, new_state, done)
        routex.append(env.get_property('lat_geod_deg'))
        routey.append(env.get_property('lng_geoc_deg'))
        if len(agent.memory) > batch_size:
            for agent in agents:
                agent.update(batch_size)        
        if (episode+1) % 150 == 0 and enable_fgfs:
            env.render()
            print('action={} state={} reward={}'.format(action_n, new_state_n, reward_n),end='\r')
        state_n = new_state_n
        episode_reward += reward_n
        if done_n[0]:
            if episode % 10 == 0: print()
            sys.stdout.write("episode: {}, reward: {}, average _reward: {:.2f} \n".
                    format(episode, np.round(episode_reward, decimals=2), 
                    np.mean(rewards[-10:])))
            plt.figure(1)
            plt.clf()
            plt.xlim(latstart-max__travel_dist, latstart+max__travel_dist)
            plt.ylim(lonstart-max__travel_dist, lonstart+max__travel_dist)
            plt.plot(routex,routey,'.')
            plt.show()
            plt.pause(0.001)        
          #  print ('target_head = {}'.format(t_head))
           # t_head += 0.1
            break
    rewards.append(episode_reward)
    avg_rewards.append(np.mean(rewards[-10:]))
    plt.figure(2)
    plt.clf()
    plt.plot(rewards)
    plt.plot(avg_rewards)

plt.ioff()
plt.show()
