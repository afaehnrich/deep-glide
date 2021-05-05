
import gym
import numpy as np
import seaborn
import torch as th
from stable_baselines3 import A2C
import deep_glide

def train_withmap_v1(env, model=None):
    if model == None:
        model = A2C(
            'MlpPolicy',
            env,
            verbose = 1,
            policy_kwargs = dict(log_std_init=-3, net_arch=[4096, 1024]),
            use_sde = True,
            buffer_size = 300000,
            batch_size = 256,
            ent_coef = 'auto',
            gamma = 0.98,
            tau = 0.02,
            train_freq = 64,
            gradient_steps = 64,
            learning_starts = 1000,
            )
    model.learn( 
        total_timesteps=1e6)
    return model

env = gym.make('JSBSim2D-v0')
model = train_withmap_v1(env)
print('Model fertig trainiert! Enter drücken!')
input()
obs = env.reset()
for i in range(5000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()
input()