import numpy as np
from deep_glide.jsbgym_new.sim import Sim, SimState, TerrainClass, TerrainOcean
from deep_glide.jsbgym_new.abstractSimHandler import AbstractJSBSimEnv, TerminationCondition
from deep_glide.jsbgym_new.properties import Properties
import logging
from deep_glide.jsbgym_new.guidance import angle_between


class JSBSimEnv_v0(AbstractJSBSimEnv): 
    metadata = {'render.modes': ['human']}

    terrain: TerrainClass = TerrainOcean()

    action_props = [Properties.custom_dir_x,
                    Properties.custom_dir_y]
    #               ,rl_wrapper.properties.Properties.custom_dir_z]

    observation_props = [Properties.attitude_psi_rad,
                        Properties.attitude_roll_rad,
                        Properties.velocities_p_rad_sec,
                        Properties.velocities_q_rad_sec,
                        Properties.velocities_r_rad_sec,
                        Properties.infinity,
                        Properties.infinity,
                        Properties.infinity,
                        Properties.infinity,
                        Properties.infinity,
                        Properties.infinity,
                        Properties.infinity,
                        Properties.infinity,
                        Properties.infinity
                        ]

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
                        self.speed[2]
                        ])
        return self.stateNormalizer.normalize(state)

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
    metadata = {'render.modes': ['human']}

    observation_props = [Properties.attitude_psi_rad,
                        Properties.attitude_roll_rad,
                        Properties.velocities_p_rad_sec,
                        Properties.velocities_q_rad_sec,
                        Properties.velocities_r_rad_sec,
                        Properties.infinity,
                        Properties.infinity,
                        Properties.infinity,
                        Properties.infinity,
                        Properties.infinity,
                        Properties.infinity,
                        Properties.infinity,
                        Properties.infinity,
                        Properties.infinity,
                        Properties.plusminusone,
                        Properties.plusminusone
                        ]

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
                        self.goal_dir[0],
                        self.goal_dir[1]
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
        self.goal_dir = np.zeros(2)
        while np.linalg.norm(self.goal_dir) ==0: 
            self.goal_dir = np.random.uniform(-1., 1., 2)
        self.goal_dir = self.goal_dir / np.linalg.norm(self.goal_dir) 
        return super().reset()

    def _reward(self):
        self._checkFinalConditions()
        if self.terminal_condition == TerminationCondition.NotFinal:
            dir_target = self.goal-self.pos
            angle = angle_between(dir_target[0:2], self.speed[0:2])
            if angle == 0: return 0.
            rew = -abs(angle_between(dir_target[0:2], self.speed[0:2]))/np.math.pi / 100.       
        elif self.terminal_condition == TerminationCondition.Arrived: 
            rew = 10. - angle_between(self.goal_dir[0:2], self.speed[0:2])/np.math.pi*5
        else:
            dist_target = np.linalg.norm(self.goal[0:2]-self.pos[0:2])
            rew = -dist_target/3000. - angle_between(self.goal_dir[0:2], self.speed[0:2])/np.math.pi*5
        if not np.isfinite(rew).all():
            logging.error('Infinite number detected in state. Replacing with zero')
            logging.error('State: {} reward: {}'.format(self._get_state(), rew))
            rew = np.nan_to_num(rew, neginf=0, posinf=0)
        return rew

    # def _reward(self, terminal_condition):
    #     dist_target = np.linalg.norm(self.obs.goal[0:2]-self.obs.pos[0:2])
    #     dist_to_ground = self.obs.pos[2] - self.terrain.altitude(self.obs.pos[0], self.obs.pos[1])
    #     #dist_max = np.linalg.norm(np.array([0,0]) - np.array(self.obs.goal[0:2]))
    #     #reward = -np.log(dist_target/dist_to_ground)
    #     reward = -np.log(dist_target/800)-dist_target/dist_to_ground
    #     if terminal_condition == TerminationCondition.Arrived: reward +=15000.
    #     if terminal_condition == TerminationCondition.Ground: reward -= 5000.
    #     elif terminal_condition == TerminationCondition.HitTerrain: reward -= 5000.
    #     elif terminal_condition == TerminationCondition.LowerThanTarget: reward -= 5000.        
    #     return reward

    # def reward_head(self, step, max_steps):
    #     dir_target = self.obs.goal-self.obs.pos
    #     #dir_target = self.obs.goal-self.obs.start
    #     v_aircraft = self.sim.get_speed_earth()
    #     reward = -abs(angle_between(dir_target[0:2], v_aircraft[0:2]))/np.math.pi
    #     done = False
    #     return self.reward_check_final(reward, done, step, max_steps)

    # def reward_distance(self, step, max_steps):
    #     dist_target = np.linalg.norm(self.obs.goal[0:2]-self.obs.pos[0:2])
    #     dist_max = np.linalg.norm(np.array([0,0]) - np.array(self.obs.goal[0:2]))
    #     reward = -np.clip(-1., 1., dist_target/dist_max)
    #     done = False
    #     return self.reward_check_final(reward, done, step, max_steps)


    # def reward_head_original(self, terminal_condition):
    #     dir_target = self.obs.goal-self.obs.pos
    #     #dir_target = self.obs.goal-self.obs.start
    #     v_aircraft = self.sim.get_speed_earth()
    #     reward = -abs(angle_between(dir_target[0:2], v_aircraft[0:2]))
    #     if terminal_condition == TerminationCondition.Arrived: reward +=50
    #     elif terminal_condition == TerminationCondition.Ground: reward = reward *100
    #     elif terminal_condition == TerminationCondition.HitTerrain: reward = reward *100
    #     elif terminal_condition == TerminationCondition.LowerThanTarget: reward = reward * 100
    #     return reward