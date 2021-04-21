from deep_glide.jsbgym_new.sim_handler_rl import JSBSimEnv_v1
import numpy as np
from deep_glide.jsbgym_new.sim import Sim, SimState, TerrainClass, TerrainOcean
from deep_glide.jsbgym_new.abstractSimHandler import AbstractJSBSimEnv, TerminationCondition
from deep_glide.jsbgym_new.properties import Properties, PropertylistToBox

import logging
from deep_glide.jsbgym_new.guidance import angle_between
from gym import spaces 


class JSBSimEnv2D_v0(JSBSimEnv_v1): 
    '''
    In diesem Env ist der Reward abhängig davon, wie nahe der Agent dem Ziel gekommen ist. 
    Höhe und Anflugwinkel sind nicht entscheidend.
    '''

    metadata = {'render.modes': ['human']}

    OBS_WIDTH = 96
    OBS_HEIGHT = 96

    terrain: TerrainClass = TerrainOcean()

    action_props = [Properties.custom_dir_x,
                    Properties.custom_dir_y]
    #               ,rl_wrapper.properties.Properties.custom_dir_z]

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6, self.OBS_WIDTH)
        )        

    def _get_state(self):
        state_float = super()._get_state()
        state_float = self.stateNormalizer.normalize(state_float)
        HUD_DIM = 6
        # state_float = np.kron(state_float, np.ones((HUD_DIM, HUD_DIM))) # konvert to 2D-"HUD"
        # state_2d = self.terrain.map_window(self.pos[0], self.pos[1], self.OBS_WIDTH, self.OBS_HEIGHT-HUD_DIM)
        # state = np.concatenate([state_2d,state_float])
        state = np.kron(state_float, np.ones((HUD_DIM, HUD_DIM))) # konvert to 2D-"HUD"
        return state

    # def _checkFinalConditions(self):
    #     if np.linalg.norm(self.goal[0:2] - self.pos[0:2])<500:
    #         logging.debug('Arrived at Target')
    #         self.terminal_condition = TerminationCondition.Arrived
    #     elif self.pos[2]<self.goal[2]-10:
    #         logging.debug('   Too low: ',self.pos[2],' < ',self.goal[2]-10)
    #         self.terminal_condition = TerminationCondition.LowerThanTarget
    #     elif self.pos[2]<=self.terrain.altitude(self.pos[0], self.pos[1])+ self.min_distance_terrain:
    #         logging.debug('   Terrain: {:.1f} <= {:.1f}+{:.1f}'.format(self.pos[2],
    #                 self.terrain.altitude(self.pos[0], self.pos[1]), self.min_distance_terrain))
    #         self.terminal_condition = TerminationCondition.HitTerrain
    #     else: self.terminal_condition = TerminationCondition.NotFinal
    #     return self.terminal_condition

    # def _done(self):
    #     self._checkFinalConditions()
    #     if self.terminal_condition == TerminationCondition.NotFinal:
    #         return False
    #     else:
    #         return True

    # def _reward(self):
    #     self._checkFinalConditions()
    #     if self.terminal_condition == TerminationCondition.NotFinal:
    #         dir_target = self.goal-self.pos
    #         v_aircraft = self.speed
    #         angle = angle_between(dir_target[0:2], v_aircraft[0:2])
    #         if angle == 0: return 0.
    #         return -abs(angle_between(dir_target[0:2], v_aircraft[0:2]))/np.math.pi / 100.       
    #     if self.terminal_condition == TerminationCondition.Arrived: return +10.
    #     dist_target = np.linalg.norm(self.goal[0:2]-self.pos[0:2])
    #     return -dist_target/3000.

