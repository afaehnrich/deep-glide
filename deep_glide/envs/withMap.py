from abc import abstractmethod
from enum import auto
from deep_glide.envs.withoutMap import JSBSimEnv_v0
import numpy as np
from deep_glide.sim import Sim, SimState, TerrainBlockworld, TerrainClass, TerrainClass90m, TerrainOcean, TerrainSingleBlocks
from deep_glide.envs.abstractEnvironments import AbstractJSBSimEnv, TerminationCondition
import deep_glide.envs.rewardFunctions as rewardFunctions
from deep_glide.deprecated.properties import Properties, PropertylistToBox
from deep_glide.utils import Normalizer, Normalizer2D
from gym.envs.registration import register

import logging
from deep_glide.utils import angle_between
from gym import spaces 
from matplotlib import pyplot as plt
import math

class AbstractJSBSimEnv2D(JSBSimEnv_v0):

    metadata = {'render.modes': ['human']}

    OBS_WIDTH = 36
    OBS_HEIGHT = 36

    observation_space: spaces.Box

    map_mean: float
    map_std: float

    def __init__(self, terrain: str, save_trajectory = False, render_before_reset=False):
        super().__init__(save_trajectory, render_before_reset)          
        self._init_terrain(terrain)
        self.observation_space = spaces.Box( low = -math.inf, high = math.inf,
                    shape=(super().observation_space.shape[0]+self.OBS_HEIGHT*self.OBS_WIDTH,), dtype=np.float32)

    def _init_terrain(self, terrain):
        if terrain == 'ocean': self.terrain = TerrainOcean()
        elif terrain == 'oceanblock': self.terrain = TerrainBlockworld(ocean=True)
        elif terrain == 'alps': self.terrain = TerrainClass90m()
        elif terrain == 'block': self.terrain = TerrainBlockworld()
        elif terrain == 'singleblock': self.terrain = TerrainSingleBlocks()
        else: raise ValueError('Terraintype unknown: {}'.format(terrain))
        print( 'using Terrain:', terrain)
        self.calc_map_mean_std()

    def calc_map_mean_std(self):
        self.map_mean = 5000.
        self.map_std = 5000.
        # (x1,x2), (y1,y2) = self.config.map_start_range
        # map_min5 = np.percentile(self.terrain.data[x1:x2, y1:y2], 5)
        # map_max5 = np.percentile(self.terrain.data[x1:x2, y1:y2], 95)
        # self.map_mean = map_min5 + (map_max5-map_min5)/2
        # self.map_std = abs((map_max5-map_min5)/2) + 0.00002
        # logging.debug('Map mean={:.2f} std={:.2f}'.format(self.map_mean, self.map_std))
        #print('Map mean={:.2f} std={:.2f}'.format(self.map_mean, self.map_std))

    def _get_state(self):
        state = super()._get_state()        
        map = self.terrain.map_around_position(self.pos[0], self.pos[1], self.OBS_WIDTH, self.OBS_HEIGHT).copy()
        map = (map-self.map_mean)/self.map_std
        #map = self.mapNormalizer.normalize(map.view().reshape(1,self.OBS_WIDTH,self.OBS_HEIGHT))
        if not np.isfinite(state).all():
            logging.error('Infinite number detected in state. Replacing with zero')
            logging.error('State: {}'.format(state))
            state = np.nan_to_num(state, neginf=0, posinf=0)
        #state = self.stateNormalizer.normalize(state.view().reshape(1,17))
        if not np.isfinite(state).all():
            logging.error('Infinite number after Normalization!')    
            raise ValueError()
        state = np.concatenate((map.flatten(), state.flatten()))
        return state


class JSBSimEnv2D_v0(AbstractJSBSimEnv2D): 

    # stateNormalizer = Normalizer('JsbSimEnv2D_v0')
    # mapNormalizer = Normalizer2D('JsbSimEnv2D_v0_map')

    env_name = 'JSBSim2D-v0'

    '''
    In diesem Env ist der Reward abhängig davon, wie nahe der Agent dem Ziel gekommen ist. 
    Höhe und Anflugwinkel sind nicht entscheidend.
    '''

    def __init__(self, terrain='ocean', save_trajectory = False, render_before_reset=False):
        super().__init__(terrain, save_trajectory, render_before_reset)


class JSBSimEnv2D_v1(JSBSimEnv2D_v0): 
    env_name = 'JSBSim2D-v1'
    '''
    Wie JSBSimEnv2D_v0, aber mit richtigen Hindernissen
    '''
    def __init__(self, terrain='block', save_trajectory = False, render_before_reset=False):
        super().__init__(terrain, save_trajectory, render_before_reset)

class JSBSimEnv2D_v2(JSBSimEnv2D_v1): 
    env_name = 'JSBSim2D-v2'

    '''
    Wie JSBSim_v5, aber mit Map.
    '''

    def __init__(self, terrain='block', save_trajectory = False, render_before_reset=False, range_dist = 500):
        super().__init__(terrain, save_trajectory, render_before_reset)
        self.RANGE_DIST = range_dist  # in m | Umkreis um das Ziel in Metern, bei dem es einen positiven Reward gibt

    _checkFinalConditions = rewardFunctions._checkFinalConditions_v5
    _reward = rewardFunctions._reward_v5

class JSBSimEnv2D_v3(JSBSimEnv2D_v2): 
    env_name = 'JSBSim2D-v3'

    '''
    Wie JSBSim_v5, aber mit Map.    
    Observation Shape angepasst für CNNs anstelle von MLPs
    '''

    def __init__(self, terrain='block', save_trajectory = False, render_before_reset=False, range_dist = 500):
        super().__init__(terrain, save_trajectory, render_before_reset, range_dist)
        self.observation_space = self.observation_space = spaces.Box(
            low=-math.inf, high=math.inf, shape=(1,37, 36)
        )


    def _get_state(self):
        # super observation : 32x32 + 15x1 --> 17x1 anhängen und dann in 33x32 umformen
        state = np.concatenate((super()._get_state(), np.zeros(21)))
        state = np.reshape(state, (1,37,36))
        return state

class JSBSimEnv2D_v4(JSBSimEnv2D_v2): 
    env_name = 'JSBSim2D-v4'

    '''
    Wie JSBSim_v6, aber mit Map.
    Ergebnis: Kein Lernen, selbst mit Ocean-Map. Wird der State auf den normalen State ohne Map reduziert, funktioniert alles super.
    '''

    _checkFinalConditions = rewardFunctions._checkFinalConditions_v6
    _reward = rewardFunctions._reward_v6

    def __init__(self, terrain='block', save_trajectory = False, render_before_reset=False, range_dist=500, range_angle = math.pi/5, angle_importance=0.5):
        super().__init__(terrain, save_trajectory, render_before_reset, range_dist)
        self.RANGE_ANGLE = range_angle  # in rad | Toleranz des Anflugwinkels, bei dem ein positiver Reward gegeben wird
        self.ANGLE_IMPORTANCE = angle_importance * 10

class JSBSimEnv2D_v5(JSBSimEnv2D_v4): 
    env_name = 'JSBSim2D-v5'
    '''
    Wie JSBSim2D-v4, aber mit nur einem Block pro episode. Der liegt dafür genau zwischen Start und Ziel

    '''

    def __init__(self, terrain='singleblock', save_trajectory = False, render_before_reset=False, range_dist=500, range_angle = math.pi/5):
        super().__init__(terrain, save_trajectory, render_before_reset, range_dist, range_angle)

    def reset(self):
        obs = super().reset()
        self.terrain.reset_map()
        self.terrain.create_block_between(self.start[:2], self.goal[0:2])
        return obs
        
# class JSBSimEnv2D_v5(JSBSimEnv2D_v2): 
#     env_name = 'JSBSim2D-v6'
#     '''
#     Wie JSBSim2D-v2, aber mit nur einem Block pro episode. Der liegt dafür genau zwischen Start und Ziel
#     (Da JSBSim2D-v4 leider nicht funktioniert)

#     '''

#     def __init__(self, terrain='singleblock', save_trajectory = False, render_before_reset=False):
#         super().__init__(terrain, save_trajectory, render_before_reset)

#     def reset(self):
#         obs = super().reset()
#         self.terrain.reset_map()
#         self.terrain.create_block_between(self.start[:2], self.goal[0:2])
#         return obs
        

register(
    id='JSBSim2D-v0',
    entry_point='deep_glide.envs.withMap:JSBSimEnv2D_v0',
    max_episode_steps=999,
    reward_threshold=1000.0,
)

register(
    id='JSBSim2D-v1',
    entry_point='deep_glide.envs.withMap:JSBSimEnv2D_v1',
    max_episode_steps=999,
    reward_threshold=1000.0,
)

register(
    id='JSBSim2D-v2',
    entry_point='deep_glide.envs.withMap:JSBSimEnv2D_v2',
    max_episode_steps=999,
    reward_threshold=1000.0,
)

register(
    id='JSBSim2D-v3',
    entry_point='deep_glide.envs.withMap:JSBSimEnv2D_v3',
    max_episode_steps=999,
    reward_threshold=1000.0,
)

register(
    id='JSBSim2D-v4',
    entry_point='deep_glide.envs.withMap:JSBSimEnv2D_v4',
    max_episode_steps=999,
    reward_threshold=1000.0,
)

register(
    id='JSBSim2D-v5',
    entry_point='deep_glide.envs.withMap:JSBSimEnv2D_v5',
    max_episode_steps=999,
    reward_threshold=1000.0,
)

register(
    id='JSBSim2D-v6',
    entry_point='deep_glide.envs.withMap:JSBSimEnv2D_v6',
    max_episode_steps=999,
    reward_threshold=1000.0,
)