import numpy as np
from deep_glide.sim import Sim, SimState, TerrainClass, TerrainOcean
from deep_glide.envs.abstractEnvironments import AbstractJSBSimEnv, TerminationCondition

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

    def __init__(self):
        initial_props={
                #'ic/h-sl-ft': 0,#3600./0.3048,
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
        state_start.position = np.array([0, 0, 3000])  #  Start Node
        state_start.props['ic/h-sl-ft'] = state_start.position[2]/0.3048

        return super().__init__(state_start, save_trajectory=False)

    '''
    Alle Environments bekommen den gleichen State, damit hinterher Transfer Learning angewendet werden kann.
    '''

    action_space = spaces.Box( low = np.array([-1., -1.]),
                              high = np.array([ 1.,  1.]) )
    observation_space = spaces.Box( low = np.array([0., -math.pi, -math.inf, -math.inf, -math.inf, -math.inf, -math.inf, 
                                           -math.inf, -math.inf, -math.inf, -math.inf, -math.inf, -math.inf, -math.inf, -1, -1]),
                                    high = np.array([2.0 * math.pi,  math.pi, -math.inf, -math.inf, -math.inf, -math.inf, -math.inf,
                                            -math.inf, -math.inf, -math.inf, -math.inf, -math.inf, -math.inf, -math.inf, 1, 1]) )

    def _get_state(self):
        state = np.array([self.sim.sim['attitude/psi-rad'],
                        self.sim.sim['attitude/roll-rad'],
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
                        self.goal_orientation[1]
                        ])
        if not np.isfinite(state).all():
            logging.error('Infinite number detected in state. Replacing with zero')
            logging.error('State: {}'.format(state))
            state = np.nan_to_num(state, neginf=0, posinf=0)
        state = self.stateNormalizer.normalize(state)
        if not np.isfinite(state).all():
            logging.error('Infinite number after Normalization!')    
            raise ValueError()
        return state

    def reset(self):
        self.goal_orientation = np.zeros(2)
        while np.linalg.norm(self.goal_orientation) ==0: 
            self.goal_orientation = np.random.uniform(-1., 1., 2)
        self.goal_orientation = self.goal_orientation / np.linalg.norm(self.goal_orientation) 
        return super().reset()

    def _checkFinalConditions(self):
        if np.linalg.norm(self.goal[0:2] - self.pos[0:2])<500:
            logging.debug('Arrived at Target')
            self.terminal_condition = TerminationCondition.Arrived
        elif self.pos[2]<self.goal[2]-10:
            logging.debug('   Too low: ',self.pos[2],' < ',self.goal[2]-10)
            self.terminal_condition = TerminationCondition.LowerThanTarget
        elif self.pos[2]<=self.terrain.altitude(self.pos[0], self.pos[1])+ self.min_distance_terrain:
            logging.debug('   Terrain: {:.1f} <= {:.1f}+{:.1f}'.format(self.pos[2],
                    self.terrain.altitude(self.pos[0], self.pos[1]), self.min_distance_terrain))
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
                rew = - dist_target / energy * 29.10
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
    '''
    Wei v3, aber mit leicht geändertem final reward
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
    '''
    In diesem Env ist der Reward abhängig davon, wie nahe der Agent dem Ziel gekommen ist.
    Der negative reward ist dabei abhängig vom Energieverlust im Verhältnis zur Entfernung zum Ziel (siehe v2).
    Der Final reward wird erst vergeben, wenn die Höhe abgebaut wurde. 
    D.h. hier wird auf jeden Fall gelandet - entweder am Ziel oder im "Gelände"
    Der Anflugwinkel am Ziel spielt keine Rolle.
    '''
    
    def _checkFinalConditions(self):
        if self.pos[2]<=self.terrain.altitude(self.pos[0], self.pos[1])+ self.min_distance_terrain:
            logging.debug('   Terrain: {:.1f} <= {:.1f}+{:.1f}'.format(self.pos[2],
                    self.terrain.altitude(self.pos[0], self.pos[1]), self.min_distance_terrain))
            self.terminal_condition = TerminationCondition.HitTerrain
        else: self.terminal_condition = TerminationCondition.NotFinal
        if self.terminal_condition != TerminationCondition.NotFinal \
           and np.linalg.norm(self.goal[0:2] - self.pos[0:2])<500:
            logging.debug('Arrived at Target')
            self.terminal_condition = TerminationCondition.Arrived
        return self.terminal_condition

class JSBSimEnv_v6(JSBSimEnv_v5):
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