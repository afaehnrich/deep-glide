import gym
from gym import envs
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import jsbgym_flex
import jsbgym_flex.properties as prp
from jsbgym_flex.environment import JsbSimEnv
from jsbgym_flex.tasks import Shaping, MyFlightTask
import pathlib
import random
import math
import toml

cfg = toml.load('gym-jsbsim-cfg.toml')

_t_head = cfg.get('environment').get('initial_state').get('target_heading')
_pid_head_p = cfg.get('pid').get('heading').get('p')
_pid_head_i = cfg.get('pid').get('heading').get('i')
_pid_head_d = cfg.get('pid').get('heading').get('d')
                

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
            if t_head!=_t_head or pid_head_p!=_pid_head_p or pid_head_i!=_pid_head_i \
                or pid_head_d!=_pid_head_d:
                env.set_property('target_heading', t_head)
                p, _, _, _ = env.pid_controls.get('heading')
                p.tune(pid_head_p, pid_head_i, pid_head_d)
                t_head = _t_head
                pid_head_p = _pid_head_p
                pid_head_i = _pid_head_i
                pid_head_d = _pid_head_d
        #if i % 5 == 0: action=env.action_space.sample()
        #action=np.array([0.8])
        action=env.action_space.sample()                
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
            print ('{:.0f}s: action={} targets[r p h]=[{:.1f} {:.1f} {:.1f}]'
                ' attitiude [r p h]=[{:.2f} {:.2f} {:.2f}] position [x y h]: [{:.0f} {:.0f} {:.0f}]          '
                .format(time, action, t_roll, t_pitch, t_head, roll, pitch, heading, lat_m, lon_m, alt_m),end='\r')
        env.step(np.array(action))
    return

env = jsbgym_flex.environment.JsbSimEnv(cfg = cfg, task_type = MyFlightTask, shaping = Shaping.STANDARD)
render = not (cfg.get('visualiser') or {}).get('enable') == False
simulate(10000, render)
