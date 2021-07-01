from deep_glide.envs.abstractEnvironments import TerminationCondition
import logging
import numpy as np
from deep_glide.utils import Normalizer, ensure_dir, angle_between

def finalConditionsDistanceOnly(self):
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

def rewardDistanceOnly(self):
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

# Rewards ohne den Final Reward bei 25 Episoden mit random actions:
# Reward not final min=-0.00999 max=-0.00000, mean=-0.00489, med=-0.00483 total per episode=-0.40984
# Der reward für den fall NotFinal wird so bemessen, dass er im Schnitt etwa -0.005 pro step beträgt.
# Ein zu großer negativer reward im NotFinal-Fall führt zu suizifalem Verhalten.
def rewardDistanceAndAngleV1(self):
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
    return rew

# Energy und 
# Energy min=61089841.11 max=115960677.29, mean=88758480.71, med=88374223.58 
# Distance min=82.33 max=11876.13, mean=5153.14, med=5060.25 
# Der reward für den fall NotFinal wird so bemessen, dass er im Schnitt etwa -0.005 pro step beträgt.
# Ein zu großer negativer reward im NotFinal-Fall führt zu suizifalem Verhalten.
# Bei steigender mittlerer Entfernung zum Ziel muss der NotFinal-reward vermutlich weiter reduziert werden,
# so dass weiterhin für random actions pro Episode ein NotFinal-reward von ca. -0.5 erzielt wird.

def rewardDistanceEnergy(self):
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
    return rew  

def _checkFinalConditions_v5(self):
    if self.pos[2]>self.terrain.altitude(self.pos[0], self.pos[1])+ self.config.min_distance_terrain:
        self.terminal_condition = TerminationCondition.NotFinal
        return self.terminal_condition
    logging.debug('   Terrain: {:.1f} <= {:.1f}+{:.1f}'.format(self.pos[2],
                self.terrain.altitude(self.pos[0], self.pos[1]), self.config.min_distance_terrain))
    if np.linalg.norm(self.goal[0:2] - self.pos[0:2])<self.RANGE_DIST:
        logging.debug('Arrived at Target')
        self.terminal_condition = TerminationCondition.Arrived
    else:
        self.terminal_condition = TerminationCondition.HitTerrain    
    return self.terminal_condition

def _checkFinalConditions_v6(self):
    '''
    Final Condition: am Boden angekommen. Die Episode ist erfolgreich, wenn Entfernung zum Ziel und Anflugwinkel stimmen
    '''
    if self.pos[2]>self.terrain.altitude(self.pos[0], self.pos[1])+ self.config.min_distance_terrain:
        self.terminal_condition = TerminationCondition.NotFinal
        return self.terminal_condition    
    logging.debug('   Terrain: {:.1f} <= {:.1f}+{:.1f}'.format(self.pos[2],
                self.terrain.altitude(self.pos[0], self.pos[1]), self.config.min_distance_terrain))
        
    if (np.linalg.norm(self.goal[0:2] - self.pos[0:2]) < self.RANGE_DIST) \
        and (abs(angle_between(self.goal_orientation[0:2], self.speed[0:2])) < self.RANGE_ANGLE) :
        logging.debug('Arrived at Target')
        self.terminal_condition = TerminationCondition.Arrived
    else:
        self.terminal_condition = TerminationCondition.HitTerrain    
    return self.terminal_condition

def _reward_v5(self):
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
    return rew  

def _reward_v6(self):    
    '''
    Der Reward ist abhängig von Entfernung zum Ziel sowie Anflugwinkel
    '''
    self._checkFinalConditions()
    rew = 0
    dist_target = np.linalg.norm(self.goal[0:2]-self.pos[0:2])
    delta_angle = abs(angle_between(self.goal_orientation[0:2], self.speed[0:2]))
    if self.terminal_condition == TerminationCondition.NotFinal:
        energy = self._get_energy()
        if energy == 0:
            rew = 0
        else:
            rew = - dist_target / energy * 29.10
    elif self.terminal_condition == TerminationCondition.Arrived: 
        rew_dist = (self.RANGE_DIST-dist_target)/self.RANGE_DIST*5
        rew_angle = (self.RANGE_ANGLE-delta_angle) / self.RANGE_ANGLE * 5
        rew = rew_angle + rew_dist
    else:
        rew_dist = min(self.RANGE_DIST-dist_target,0)/3000/1.3
        rew_angle = min(self.RANGE_ANGLE-delta_angle,0)
        rew = rew_angle + rew_dist       
    return rew  
