from enum import auto
import numpy as np
from deep_glide.sim import Sim, SimState, TerrainBlockworld, TerrainClass, TerrainClass90m, TerrainOcean
from deep_glide.envs.abstractEnvJSBSimRaw import AbstractJSBSimRawEnv
from deep_glide.envs.abstractEnvironments import TerminationCondition
from deep_glide.utils import Normalizer, ensure_dir, angle_between
from gym.envs.registration import register

import logging
from gym import spaces 
import math
from matplotlib import pyplot as plt
from scipy.interpolate import UnivariateSpline
import os
from datetime import date

class JSBSimRaw_v0(AbstractJSBSimRawEnv):
    env_name = 'JSBSimRaw-v0'
    '''
    In diesem Env ist der Reward abhängig davon, wie nahe der Agent dem Ziel gekommen ist. 
    Höhe und Anflugwinkel sind nicht entscheidend.
    '''

    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()
        self.terrain = TerrainOcean()

    '''
    Alle Environments bekommen den gleichen State, damit hinterher Transfer Learning angewendet werden kann.
    '''

    def reset(self):
        state = super().reset()
        self.old_energy = self._get_energy()
        return state

    def _checkFinalConditions(self):
        if np.linalg.norm(self.goal[0:2] - self.pos[0:2])<500:
            logging.debug('Arrived at Target')
            self.terminal_condition = TerminationCondition.Arrived
        elif self.pos[2]<self.goal[2]-10:
            logging.debug('   Too low: ',self.pos[2],' < ',self.goal[2]-10)
            self.terminal_condition = TerminationCondition.LowerThanTarget
        elif self.pos[2]<=self.terrain.altitude(self.pos[0], self.pos[1])+ self.config.min_distance_terrain:
            logging.debug('   Terrain: {:.1f} <= {:.1f}+{:.1f}'.format(self.pos[2],
                    self.terrain.altitude(self.pos[0], self.pos[1]), self.config.min_distance_terrain))
            self.terminal_condition = TerminationCondition.HitTerrain
        else: self.terminal_condition = TerminationCondition.NotFinal
        return self.terminal_condition

    def _done(self):
        self._checkFinalConditions()
        if self.terminal_condition == TerminationCondition.NotFinal:
            return False
        else:
            return True

    def _reward(self):
        #energy = self._get_energy()
        #rew = (energy - self.old_energy) / 88758480.71 
        #rew = np.nan_to_num(rew, neginf=0, posinf=0)
        #self.old_energy = energy
        rew = 1/1000
        return rew


register(
    id='JSBSimRaw-v0',
    entry_point='deep_glide.envs.rawInputs:JSBSimRaw_v0',
    max_episode_steps=999999,
    reward_threshold=100000.0,
)
