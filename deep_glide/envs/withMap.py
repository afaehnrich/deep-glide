from abc import abstractmethod
from enum import auto
from deep_glide.envs.withoutMap import Scenario_A
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
import os

class AbstractJSBSimEnv2D(Scenario_A):

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


class Scenario_A_Terrain(AbstractJSBSimEnv2D): 

    # stateNormalizer = Normalizer('JsbSimEnv2D_v0')
    # mapNormalizer = Normalizer2D('JsbSimEnv2D_v0_map')

    env_name = 'Scenario_A_Terrain-v0'

    '''
    In diesem Env ist der Reward abhängig davon, wie nahe der Agent dem Ziel gekommen ist. 
    Höhe und Anflugwinkel sind nicht entscheidend.
    '''

    def __init__(self, terrain='ocean', save_trajectory = False, render_before_reset=False):
        super().__init__(terrain, save_trajectory, render_before_reset)


class Scenario_B_Terrain(Scenario_A_Terrain): 
    env_name = 'Scenario_B_Terrain-v0'

    '''
    Wie JSBSim_v5, aber mit Map.
    '''

    def __init__(self, terrain='ocean', save_trajectory = False, render_before_reset=False, range_dist = 500, goto_time = 5.):
        super().__init__(terrain, save_trajectory, render_before_reset)
        self.RANGE_DIST = range_dist  # in m | Umkreis um das Ziel in Metern, bei dem es einen positiven Reward gibt
        self.goto_time = goto_time
        # Aktivieren, wenn mehr Logging benötigt wird:
        # self.log_fn = 'Log_JSBSim2D-v2_final_heights'
        # i=1
        # while os.path.exists('{}_{}.csv'.format(self.log_fn, i)): i+=1
        # self.log_fn = '{}_{}.csv'.format(self.log_fn, i)
        # with open(self.log_fn,'w') as fd:
        #     fd.write('height; terrain_height\n')  


    def step(self, action):
        new_state, reward, done, info = super().step(action)
        # Aktivieren, wenn mehr Logging benötigt wird:
        # if done:
        #     with open(self.log_fn,'a') as fd:
        #         fd.write('{:f}; {:f}\n'.format(self.pos[2],self.terrain.altitude(self.pos[0], self.pos[1])).replace('.',','))
        return new_state, reward, done, info

    _checkFinalConditions = rewardFunctions._checkFinalConditions_v5
    _reward = rewardFunctions._reward_v5

class Scenario_C_Terrain(Scenario_B_Terrain): 
    env_name = 'Scenario_C_Terrain-v0'

    '''
    Wie JSBSim_v6, aber mit Map.
    Ergebnis: Kein Lernen, selbst mit Ocean-Map. Wird der State auf den normalen State ohne Map reduziert, funktioniert alles super.
    '''

    _checkFinalConditions = rewardFunctions._checkFinalConditions_v6
    _reward = rewardFunctions._reward_v6

    def __init__(self, terrain='ocean', save_trajectory = False, render_before_reset=False, range_dist=500, range_angle = math.pi/5, angle_importance=0.5):
        super().__init__(terrain, save_trajectory, render_before_reset, range_dist)
        self.RANGE_ANGLE = range_angle  # in rad | Toleranz des Anflugwinkels, bei dem ein positiver Reward gegeben wird
        self.ANGLE_IMPORTANCE = angle_importance * 10

        

register(
    id='Scenario_A_Terrain-v0',
    entry_point='deep_glide.envs.withMap:Scenario_A_Terrain',
    max_episode_steps=999,
    reward_threshold=1000.0,
)

register(
    id='Scenario_B_Terrain-v0',
    entry_point='deep_glide.envs.withMap:Scenario_B_Terrain',
    max_episode_steps=999,
    reward_threshold=1000.0,
)

register(
    id='Scenario_C_Terrain-v0',
    entry_point='deep_glide.envs.withMap:Scenario_C_Terrain',
    max_episode_steps=999,
    reward_threshold=1000.0,
)