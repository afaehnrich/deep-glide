import numpy as np
from deep_glide.sim import Sim, SimState, TerrainClass, TerrainOcean
from deep_glide.envs.abstractEnvironments import AbstractJSBSimEnv, TerminationCondition
from deep_glide.utils import Normalizer
from gym.envs.registration import register

import logging
from deep_glide.utils import angle_between
from gym import spaces 
import math

class JSBSimEnv_v0(AbstractJSBSimEnv): 
    '''
    In diesem Env ist der Reward abhängig davon, wie nahe der Agent dem Ziel gekommen ist. 
    Höhe und Anflugwinkel sind nicht entscheidend.
    '''

    metadata = {'render.modes': ['human']}

    terrain: TerrainClass = TerrainOcean()
    stateNormalizer = Normalizer('JsbSimEnv_v0')

    '''
    Alle Environments bekommen den gleichen State, damit hinterher Transfer Learning angewendet werden kann.
    '''

    action_space = spaces.Box( low = -1., high = 1., shape=(3,), dtype=np.float32)
    observation_space = spaces.Box( low = -math.inf, high = math.inf, shape=(17,), dtype=np.float32)

    def __init__(self):
        super().__init__()
        self.stateNormalizer.count=350000
        self.stateNormalizer.mean=[ 2.97364472e-03,  1.10485485e-02, -1.76071912e-04,  3.13598366e+02,
                                    1.63658909e+02,  1.34970980e+03, -7.85394264e+01, -3.04544640e+01,
                                    9.99997171e+01, -2.77013138e-01,  3.79266207e-01, -1.28894164e+01,
                                    -6.26786425e-03,  6.33990049e-03,  2.85714286e-06,  2.85714286e-06,
                                    2.85714286e-06]
        self.stateNormalizer.M2=[2.62146647e+03, 9.82602879e+02, 6.72278526e+02, 7.06071626e+12,
                                7.77554241e+12, 2.80579783e+11, 2.88355302e+12, 2.94527236e+12,
                                9.80197200e+03, 2.07188799e+09, 2.18822997e+09, 3.63861497e+07,
                                1.73373913e+05, 1.76601269e+05, 1.99999714e+00, 1.99999714e+00,
                                1.99999714e+00]

    def _get_state(self):
        wind = self.sim.get_wind()
        state = np.array([#self.sim.sim['attitude/psi-rad'],
                        #self.sim.sim['attitude/roll-rad'],
                        self.sim.sim['velocities/p-rad_sec'],
                        self.sim.sim['velocities/q-rad_sec'],
                        self.sim.sim['velocities/r-rad_sec'],
                        self.pos[0],
                        self.pos[1],
                        self.pos[2],
                        self.goal[0],
                        self.goal[1],
                        self.goal[2],
                        self.speed[0],
                        self.speed[1],
                        self.speed[2],
                        self.goal_orientation[0],
                        self.goal_orientation[1],
                        wind[0],
                        wind[1],
                        wind[2],
                        ])
        if not np.isfinite(state).all():
            logging.error('Infinite number detected in state. Replacing with zero')
            logging.error('State: {}'.format(state))
            state = np.nan_to_num(state, neginf=0, posinf=0)
        state = self.stateNormalizer.normalize(state.view().reshape(1,17))
        if not np.isfinite(state).all():
            logging.error('Infinite number after Normalization!')    
            raise ValueError()
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
        self._checkFinalConditions()
        if self.terminal_condition == TerminationCondition.NotFinal:
            dir_target = self.goal-self.pos
            v_aircraft = self.speed
            angle = angle_between(dir_target[0:2], v_aircraft[0:2])
            if angle == 0: return 0.
            return -abs(angle_between(dir_target[0:2], v_aircraft[0:2]))/np.math.pi / 100.       
        if self.terminal_condition == TerminationCondition.Arrived: return +10.
        dist_target = np.linalg.norm(self.goal[0:2]-self.pos[0:2])
        return -dist_target/3000.


class JSBSimEnv_v1(JSBSimEnv_v0): 

    stateNormalizer = Normalizer('JsbSimEnv_v1')

    '''
    In diesem Env ist der Reward abhängig davon, wie nahe der Agent dem Ziel gekommen ist und in welchem Winkel zum Ziel die Ankunft erfolgte.
    Die Anflughöhe wird nicht bewertet.
    '''
    
    # Rewards ohne den Final Reward bei 25 Episoden mit random actions:
    # Reward not final min=-0.00999 max=-0.00000, mean=-0.00489, med=-0.00483 total per episode=-0.40984
    # Der reward für den fall NotFinal wird so bemessen, dass er im Schnitt etwa -0.005 pro step beträgt.
    # Ein zu großer negativer reward im NotFinal-Fall führt zu suizifalem Verhalten.
    def _reward(self):
        self._checkFinalConditions()
        if self.terminal_condition == TerminationCondition.NotFinal:
            dir_target = self.goal-self.pos
            angle = angle_between(dir_target[0:2], self.speed[0:2])
            if angle == 0: return 0.
            rew = -abs(angle_between(dir_target[0:2], self.speed[0:2]))/np.math.pi / 100.       
        elif self.terminal_condition == TerminationCondition.Arrived: 
            rew = 10. - abs(angle_between(self.goal_orientation[0:2], self.speed[0:2])/np.math.pi*5)
        else:
            dist_target = np.linalg.norm(self.goal[0:2]-self.pos[0:2])
            rew = -dist_target/3000. - abs(angle_between(self.goal_orientation[0:2], self.speed[0:2])/np.math.pi*5)
        if not np.isfinite(rew).all():
            logging.error('Infinite number detected in state. Replacing with zero')
            logging.error('State: {} reward: {}'.format(self._get_state(), rew))
            rew = np.nan_to_num(rew, neginf=0, posinf=0)
        return rew

class JSBSimEnv_v2(JSBSimEnv_v1): 

    stateNormalizer = Normalizer('JsbSimEnv_v2')

    '''
    In diesem Env ist der Reward abhängig davon, wie nahe der Agent dem Ziel gekommen ist.
    Im Unterschied zu v0 wird als Zwischen-Reward jedoch nicht die Abweichung Flugwinkel - Winkel zum Ziel bewertet,
    sondern der Energieverlust im Verhältnis zur Entfernung zum Ziel.
    Dies ist eine Vorbereitung auf den späteren Anwendungsfall - das Ziel sollte möglichst 
    schnell mit möglichst wenig Energieverlust erreicht werden.
    Höhe und Anflugwinkel am Ziel spielen keine Rolle.
    '''
    # Energy und 
    # Energy min=61089841.11 max=115960677.29, mean=88758480.71, med=88374223.58 
    # Distance min=82.33 max=11876.13, mean=5153.14, med=5060.25 
    # Der reward für den fall NotFinal wird so bemessen, dass er im Schnitt etwa -0.005 pro step beträgt.
    # Ein zu großer negativer reward im NotFinal-Fall führt zu suizifalem Verhalten.
    # Bei steigender mittlerer Entfernung zum Ziel muss der NotFinal-reward vermutlich weiter reduziert werden,
    # so dass weiterhin für random actions pro Episode ein NotFinal-reward von ca. -0.5 erzielt wird.
 
    def _reward(self):
        self._checkFinalConditions()
        rew = 0
        if self.terminal_condition == TerminationCondition.NotFinal:
            dist_target = np.linalg.norm(self.goal[0:2]-self.pos[0:2])
            energy = self._get_energy()
            if energy == 0:
                rew = 0
            else:
                rew = - dist_target / energy * 29.10
        elif self.terminal_condition == TerminationCondition.Arrived: 
            rew = 10.#  - abs(angle_between(self.goal_dir[0:2], self.speed[0:2])/np.math.pi*5)
        else:
            dist_target = np.linalg.norm(self.goal[0:2]-self.pos[0:2])
            rew = -dist_target/3000.# - angle_between(self.goal_dir[0:2], self.speed[0:2])/np.math.pi*5
        if not np.isfinite(rew).all():
            logging.error('Infinite number detected in state. Replacing with zero')
            logging.error('State: {} reward: {}'.format(self._get_state(), rew))
            rew = np.nan_to_num(rew, neginf=0, posinf=0)
        return rew  

class JSBSimEnv_v3(JSBSimEnv_v1): 

    stateNormalizer = Normalizer('JsbSimEnv_v3')

    '''
    In diesem Env ist der Reward abhängig davon, wie nahe der Agent dem Ziel gekommen ist.
    Im Unterschied zu v0 wird als Zwischen-Reward jedoch nicht die Abweichung Flugwinkel - Winkel zum Ziel bewertet,
    sondern der Energieverlust im Verhältnis zur Entfernung zum Ziel.
    Dies ist eine Vorbereitung auf den späteren Anwendungsfall - das Ziel sollte möglichst 
    schnell mit möglichst wenig Energieverlust erreicht werden.
    Der korrekte Anflugwinkel wird im Final reward belohnt.
    Die Höhe spielt keine Rolle.
    '''
    
    
    def _reward(self):
        self._checkFinalConditions()
        rew = 0
        if self.terminal_condition == TerminationCondition.NotFinal:
            dist_target = np.linalg.norm(self.goal[0:2]-self.pos[0:2])
            energy = self._get_energy()
            if energy == 0:
                rew = 0
            else:
                rew = - abs(dist_target / energy * 29.10)
            if energy < 0:
                logging.error('Negative Energy! pos={} speed={} e={}'.format(self.pos, self.speed, energy))
                exit()
            if dist_target < 0:
                logging.error('Negative Distance! pos={} goal={} dist={}'.format(self.pos, self.goal, dist_target))
                exit()
        elif self.terminal_condition == TerminationCondition.Arrived: 
            rew = 10. - abs(angle_between(self.goal_orientation[0:2], self.speed[0:2])/np.math.pi*5)
        else:
            dist_target = np.linalg.norm(self.goal[0:2]-self.pos[0:2])
            rew = -dist_target/3000. - abs(angle_between(self.goal_orientation[0:2], self.speed[0:2])/np.math.pi*5)
        if not np.isfinite(rew).all():
            logging.error('Infinite number detected in state. Replacing with zero')
            logging.error('State: {} reward: {}'.format(self._get_state(), rew))
            rew = np.nan_to_num(rew, neginf=0, posinf=0)
        return rew  

class JSBSimEnv_v4(JSBSimEnv_v3): 

    stateNormalizer = Normalizer('JsbSimEnv_v4')

    '''
    Wie v3, aber mit leicht geändertem final reward
    '''
       
    def _reward(self):
        self._checkFinalConditions()
        rew = 0
        if self.terminal_condition == TerminationCondition.NotFinal:
            dist_target = np.linalg.norm(self.goal[0:2]-self.pos[0:2])
            energy = self._get_energy()
            if energy == 0:
                rew = 0
            else:
                rew = - dist_target / energy * 29.10
        elif self.terminal_condition == TerminationCondition.Arrived: 
            rew = 10. - abs(angle_between(self.goal_orientation[0:2], self.speed[0:2])/np.math.pi)*15.
        else:
            dist_target = np.linalg.norm(self.goal[0:2]-self.pos[0:2])
            rew = -dist_target/3000. - abs(angle_between(self.goal_orientation[0:2], self.speed[0:2])/np.math.pi)*15.
        if not np.isfinite(rew).all():
            logging.error('Infinite number detected in state. Replacing with zero')
            logging.error('State: {} reward: {}'.format(self._get_state(), rew))
            rew = np.nan_to_num(rew, neginf=0, posinf=0)
        return rew  


class JSBSimEnv_v5(JSBSimEnv_v2): 

    stateNormalizer = Normalizer('JsbSimEnv_v5')

    '''
    In diesem Env ist der Reward abhängig davon, wie nahe der Agent dem Ziel gekommen ist.
    Der negative reward ist dabei abhängig vom Energieverlust im Verhältnis zur Entfernung zum Ziel (siehe v2).
    Der Final reward wird erst vergeben, wenn die Höhe abgebaut wurde. 
    D.h. hier wird auf jeden Fall gelandet - entweder am Ziel oder im "Gelände"
    Der Anflugwinkel am Ziel spielt keine Rolle.
    '''
    
    # observation_space = spaces.Box( low = np.array([0., -math.pi,  -2 * math.pi, -2 * math.pi, -2 * math.pi, -math.inf, -math.inf, 
    #                                        -math.inf, -math.inf, -math.inf, -math.inf, -math.inf, -math.inf, -math.inf]),
    #                                 high = np.array([2.0 * math.pi,  math.pi,  2 * math.pi, 2 * math.pi, 2 * math.pi, math.inf, math.inf,
    #                                         math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf]) )
    # def _get_state(self):
    #     state = np.array([self.sim.sim['attitude/psi-rad'],
    #                     self.sim.sim['attitude/roll-rad'],
    #                     self.sim.sim['velocities/p-rad_sec'],
    #                     self.sim.sim['velocities/q-rad_sec'],
    #                     self.sim.sim['velocities/r-rad_sec'],
    #                     self.pos[0],
    #                     self.pos[1],
    #                     self.pos[2],
    #                     self.goal[0],
    #                     self.goal[1],
    #                     self.goal[2],
    #                     self.speed[0],
    #                     self.speed[1],
    #                     self.speed[2],
    #                     #self.goal_orientation[0],
    #                     #self.goal_orientation[1]
    #                     ])
    #     if not np.isfinite(state).all():
    #         logging.error('Infinite number detected in state. Replacing with zero')
    #         logging.error('State: {}'.format(state))
    #         state = np.nan_to_num(state, neginf=0, posinf=0)
    #     state = self.stateNormalizer.normalize(state)
    #     if not np.isfinite(state).all():
    #         logging.error('Infinite number after Normalization!')    
    #         raise ValueError()
    #     return state
        

    def _checkFinalConditions(self):
        if self.pos[2]<=self.terrain.altitude(self.pos[0], self.pos[1])+ self.config.min_distance_terrain:
            logging.debug('   Terrain: {:.1f} <= {:.1f}+{:.1f}'.format(self.pos[2],
                    self.terrain.altitude(self.pos[0], self.pos[1]), self.config.min_distance_terrain))
            self.terminal_condition = TerminationCondition.HitTerrain
        else: self.terminal_condition = TerminationCondition.NotFinal
        if self.terminal_condition != TerminationCondition.NotFinal \
           and np.linalg.norm(self.goal[0:2] - self.pos[0:2])<500:
            logging.debug('Arrived at Target')
            self.terminal_condition = TerminationCondition.Arrived
        return self.terminal_condition

class JSBSimEnv_v6(JSBSimEnv_v5):

    stateNormalizer = Normalizer('JsbSimEnv_v6')

    '''
    Dieses Env kombiniert v5 (Reward nur, wenn am Boden angekommen) 
    mit v4 (Final reward abhängig vom Anflugwinkel)
    '''

    def _reward(self):
        self._checkFinalConditions()
        rew = 0
        if self.terminal_condition == TerminationCondition.NotFinal:
            dist_target = np.linalg.norm(self.goal[0:2]-self.pos[0:2])
            energy = self._get_energy()
            if energy == 0:
                rew = 0
            else:
                rew = - dist_target / energy * 29.10
        elif self.terminal_condition == TerminationCondition.Arrived: 
            rew = 10. - abs(angle_between(self.goal_orientation[0:2], self.speed[0:2])/np.math.pi)*15.
        else:
            dist_target = np.linalg.norm(self.goal[0:2]-self.pos[0:2])
            rew = -dist_target/3000. - abs(angle_between(self.goal_orientation[0:2], self.speed[0:2])/np.math.pi)*15.
        if not np.isfinite(rew).all():
            logging.error('Infinite number detected in state. Replacing with zero')
            logging.error('State: {} reward: {}'.format(self._get_state(), rew))
            rew = np.nan_to_num(rew, neginf=0, posinf=0)
        return rew  

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
