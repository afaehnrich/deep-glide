from enum import auto
from deep_glide.envs.withoutMap import JSBSimEnv_v0
import numpy as np
from deep_glide.sim import Sim, SimState, TerrainBlockworld, TerrainClass, TerrainOcean
from deep_glide.envs.abstractEnvironments import AbstractJSBSimEnv, TerminationCondition
from deep_glide.deprecated.properties import Properties, PropertylistToBox
from deep_glide.utils import Normalizer, Normalizer2D
from gym.envs.registration import register

import logging
from deep_glide.utils import angle_between
from gym import spaces 
from matplotlib import pyplot as plt
import math


class JSBSimEnv2D_v0(JSBSimEnv_v0): 

    # stateNormalizer = Normalizer('JsbSimEnv2D_v0')
    # mapNormalizer = Normalizer2D('JsbSimEnv2D_v0_map')

    env_name = 'JSBSim2D-v0'

    '''
    In diesem Env ist der Reward abhängig davon, wie nahe der Agent dem Ziel gekommen ist. 
    Höhe und Anflugwinkel sind nicht entscheidend.
    '''

    metadata = {'render.modes': ['human']}

    OBS_WIDTH = 32
    OBS_HEIGHT = 32
    z_range = (1000, 1500)

    terrain: TerrainClass
    observation_space: spaces.Box

    map_mean: float
    map_std: float

    def __init__(self):
        super().__init__()                
        self._init_terrain()
        self.observation_space = spaces.Box( low = -math.inf, high = math.inf,
                    shape=(super().observation_space.shape[0]+self.OBS_HEIGHT*self.OBS_WIDTH,), dtype=np.float32)

    def _init_terrain(self):
        self.terrain = TerrainOcean()
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

    plot_fig: plt.figure = None


class JSBSimEnv2D_v1(JSBSimEnv2D_v0): 
    env_name = 'JSBSim2D-v1'
    '''
    Wie JSBSimEnv2D_v0, aber mit richtigen Hindernissen
    '''
    def _init_terrain(self):
        self.terrain = TerrainBlockworld()
        self.calc_map_mean_std()

class JSBSimEnv2D_v2(JSBSimEnv2D_v1): 
    env_name = 'JSBSim2D-v2'

    '''
    Wie JSBSim_v5, aber mit Map.
    '''

    def _init_terrain(self):
        self.terrain = TerrainBlockworld()
        self.calc_map_mean_std()


    RANGE_DIST = 500 # in m | Umkreis um das Ziel in Metern, bei dem es einen positiven Reward gibt
    
    def _checkFinalConditions(self):
        if self.pos[2]<=self.terrain.altitude(self.pos[0], self.pos[1])+ self.config.min_distance_terrain:
            logging.debug('   Terrain: {:.1f} <= {:.1f}+{:.1f}'.format(self.pos[2],
                    self.terrain.altitude(self.pos[0], self.pos[1]), self.config.min_distance_terrain))
            self.terminal_condition = TerminationCondition.HitTerrain
        else: self.terminal_condition = TerminationCondition.NotFinal
        if self.terminal_condition != TerminationCondition.NotFinal \
           and np.linalg.norm(self.goal[0:2] - self.pos[0:2])<self.RANGE_DIST:
            logging.debug('Arrived at Target')
            self.terminal_condition = TerminationCondition.Arrived
        return self.terminal_condition

    def _reward(self):
        self._checkFinalConditions()
        rew = 0
        dist_target = np.linalg.norm(self.goal[0:2]-self.pos[0:2])
        if self.terminal_condition == TerminationCondition.NotFinal:            
            energy = self._get_energy()
            if energy == 0:
                rew = 0
            else:
                rew = - dist_target / energy * 29.10
        elif self.terminal_condition == TerminationCondition.Arrived: 
            rew = (self.RANGE_DIST-dist_target)/self.RANGE_DIST*10
        else:
            rew = min(self.RANGE_DIST-dist_target,0)/3000
        if not np.isfinite(rew).all():
            logging.error('Infinite number detected in state. Replacing with zero')
            logging.error('State: {} reward: {}'.format(self._get_state(), rew))
            rew = np.nan_to_num(rew, neginf=0, posinf=0)
        return rew  

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

