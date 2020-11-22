import gym
from gym import envs
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import gym_jsbsim_simple
import gym_jsbsim_simple.properties as prp
from gym_jsbsim_simple.environment import JsbSimEnv
from gym_jsbsim_simple.tasks import Shaping, MyFlightTask
import pathlib
import random
import math
import toml

cfg = toml.load('gym-jsbsim-cfg.toml')



def simulate(steps, render = False):
    env.reset()
    action=[0.0]   
    for i in range(steps):
        if render: 
            env.render()
            if i%2 ==0:
                cfg = toml.load('gym-jsbsim-cfg.toml')
                t_head = cfg.get('environment').get('initial_state').get('target_heading')
                pid_head_p = cfg.get('pid').get('heading').get('p')
                pid_head_i = cfg.get('pid').get('heading').get('i')
                pid_head_d = cfg.get('pid').get('heading').get('d')
                env.set_property('target_heading', t_head)
                p, _, _, _ = env.pid_controls.get('heading')
                p.tune(pid_head_p, pid_head_i, pid_head_d)
            if i%1 ==0:
                time = env.sim[prp.sim_time_s]
                t_roll= env.sim[env.properties['target_roll']]
                t_pitch= env.sim[env.properties['target_pitch']]
                t_head= env.sim[env.properties['target_heading']]
                lat_m = env.get_property('dist_travel_lat_m')
                lon_m = env.get_property('dist_travel_lon_m')
                alt_m = env.get_property('altitude_sl_m')
                roll = env.get_property('roll_rad')
                pitch = env.get_property('pitch_rad')
                heading = env.get_property('heading_rad')
                print ('{:.0f}s: t_roll={:.1f} t_pitch={:.1f} t_head={:.1f}'
                    ' roll={:.2f}, pitch={:.2f}, heading={:.2f} position ab Start(x,y,h): [{:.0f},{:.0f},{:.0f}]          '
                    .format(time,t_roll, t_pitch, t_head, roll, pitch, heading, lat_m, lon_m, alt_m),end='\r')
        #if i % 5 == 0: action=env.action_space.sample()
        action=np.zeros(1)
        env.step(np.array(action))
    return

env = gym_jsbsim_simple.environment.JsbSimEnv(cfg = cfg, task_type = MyFlightTask, shaping = Shaping.STANDARD)
render = not (cfg.get('visualiser') or {}).get('enable') == False
simulate(100000, render)
