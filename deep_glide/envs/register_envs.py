#from deep_glide.envs.withoutMap import JSBSimEnv_v0, JSBSimEnv_v1, JSBSimEnv_v2, JSBSimEnv_v3, JSBSimEnv_v4, JSBSimEnv_v5, JSBSimEnv_v6
#from deep_glide.envs.withMap import JSBSimEnv2D_v0
from gym.envs.registration import register

def register_jsbsim_envs():
    register(
        id='JSBSim-v0',
        entry_point='deep_glide.envs.withoutMap:JSBSimEnv_v0',
        max_episode_steps=999,
        reward_threshold=1000.0,
    )
    register(
        id='JSBSim-v1',
        entry_point='deep_glide.envs.withoutMap:JSBSimEnv_v1',
        max_episode_steps=999,
        reward_threshold=1000.0,
    )
    register(
        id='JSBSim-v2',
        entry_point='deep_glide.envs.withoutMap:JSBSimEnv_v2',
        max_episode_steps=999,
        reward_threshold=1000.0,
    )
    register(
        id='JSBSim-v3',
        entry_point='deep_glide.envs.withoutMap:JSBSimEnv_v3',
        max_episode_steps=999,
        reward_threshold=1000.0,
    )
    register(
        id='JSBSim-v4',
        entry_point='deep_glide.envs.withoutMap:JSBSimEnv_v4',
        max_episode_steps=999,
        reward_threshold=1000.0,
    )
    register(
        id='JSBSim-v5',
        entry_point='deep_glide.envs.withoutMap:JSBSimEnv_v5',
        max_episode_steps=999,
        reward_threshold=1000.0,
    )
    register(
        id='JSBSim-v6',
        entry_point='deep_glide.envs.withoutMap:JSBSimEnv_v6',
        max_episode_steps=999,
        reward_threshold=1000.0,
    )
    register(
        id='JSBSim2D-v0',
        entry_point='deep_glide.envs.withMap:JSBSimEnv2D_v0',
        max_episode_steps=999,
        reward_threshold=1000.0,
    )

