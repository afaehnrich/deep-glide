import numpy as np
from deep_glide.jsbgym_new.sim import Sim, SimState, TerrainClass, TerrainOcean
from deep_glide.jsbgym_new.abstractSimHandler import AbstractJSBSimEnv
from deep_glide.jsbgym_new.properties import Properties

class SimHandlerRL(AbstractJSBSimEnv): 
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
                        Properties.infinity,
                        Properties.plusminusone,
                        Properties.plusminusone]

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
