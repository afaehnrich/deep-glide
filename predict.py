import gym

from deep_glide.jsbgym_new.sim_handler_rl import JSBSimEnv_v0
from deep_glide.jsbgym_new.sim import SimState
import numpy as np

from stable_baselines3 import A2C, PPO


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
state_start.position = np.array([0,0,3000]) # Start Node
state_start.props['ic/h-sl-ft'] = state_start.position[2]/0.3048

env = JSBSimEnv_v0(state_start, save_trajectory=False)
#env = gym.make("CarRacing-v0")
print(env.action_space)
print(env.observation_space)
model = A2C("MlpPolicy", env, verbose=1)
model.load("jsbsim_model")
obs = env.reset()
max_steps = 100000
episode_reward = 0
for i in range(max_steps):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    episode_reward += reward
    if done:
        print('Step', i,'/',max_steps,' Episode Reward: {:.2f}'.format(episode_reward))
        env.render()
        obs = env.reset()
        episode_reward = 0
env.render()
print('Fertig!')
input()
env.close()