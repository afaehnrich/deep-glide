import deep_glide.jsbgym_new.sim_handler_rl
from gym.envs.registration import register

def register_jsbsim_envs():
    register(
        id='JSBSim-v0',
        entry_point='deep_glide.jsbgym_new.sim_handler_rl:JSBSimEnv_v0',
        max_episode_steps=999,
        reward_threshold=1000.0,
    )
    register(
        id='JSBSim-v1',
        entry_point='deep_glide.jsbgym_new.sim_handler_rl:JSBSimEnv_v1',
        max_episode_steps=999,
        reward_threshold=1000.0,
    )

