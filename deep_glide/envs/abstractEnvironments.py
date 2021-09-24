from typing import Tuple, List
import numpy as np
from deep_glide.pid import PID, PID_angle
from deep_glide.sim import Sim, SimState, TerrainClass, SimTimer, TerrainOcean, Runway
from deep_glide.utils import Normalizer, angle_between, ensure_dir, ensure_newfile, rotate_vect_2d
from enum import Enum
import gym
from gym import spaces
import logging
from deep_glide import plotting
from abc import ABC, abstractmethod
from dataclasses import dataclass
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
import math
import os
from datetime import date
import time
import torch

class TerminationCondition(Enum):
    NotFinal = 0
    Arrived = 1
    LowerThanTarget = 3
    HitTerrain = 4
    Ground = 5
    OutOfBounds = 6

@dataclass
class Config:
    start_ground_distance = (1000,4000)
    goal_ground_distance = (50,50)
    speed_range = (80,100)
    x_range_start = (-5000, 5000)
    y_range_start = (-5000, 5000)    
    # x_range_start = (0, 0)
    # y_range_start = (0, 0)
    z_range_start = (0, 8000)  
    x_range_goal = (-5000, 5000)
    y_range_goal = (-5000, 5000)    
    z_range_goal = (0, 4000)
    # x_search_range = (-5000, 5000)
    # y_search_range = (-5000, 5000)        
    x_range_wind = (-30., 30.)
    y_range_wind = (-30., 30.)
    z_range_wind = (-30., 30.)
    # map_start_range =( (600,3000), (600, 3000)) # for 30m hgt
    map_start_range =( (4200,5400), (2400, 3600)) # for 90m hgt srtm_38_03.hgt
    render_range =( (-7500, 7500), (-7500, 7500)) # for 90m hgt srtm_38_03.hgt
    min_distance_terrain = 50
    ground_distance_radius = 900
    runway_dimension = np.array([900,60]) # Landebahn Länge x Breite
    runway_draw = False
    initial_props={
        'ic/terrain-elevation-ft': 0.00000001, # 0.0 erzeugt wohl NaNs
        'ic/p-rad_sec': 0,
        'ic/q-rad_sec': 0,
        'ic/r-rad_sec': 0,
        'ic/v-fps': 0,
        'ic/w-fps': 0,
        'ic/roc-fpm': 0,
        'gear/gear-pos-norm': 0.0, # landing gear raised
        'gear/gear-cmd-norm': 0.0, # lnding gear raised
        'propulsion/set-running': 0, # 1 = running; 0 = off
        'fcs/throttle-cmd-norm': 0.0, # 0.8
        'fcs/mixture-cmd-norm': 0.0, # 0.8
    }
    logdir = './logs'

class AbstractJSBSimEnv(gym.Env, ABC):
   
    metadata = {'render.modes': ['human']}
    action_space = spaces.Box( low = -1., high = 1., shape=(3,), dtype=np.float32)
    observation_space = spaces.Box( low = -math.inf, high = math.inf, shape=(15,), dtype=np.float32)
    env_name: str

    @abstractmethod
    def _checkFinalConditions(self):
        pass

    @abstractmethod
    def _done(self):
        pass

    @abstractmethod
    def _reward(self):
        pass

    @abstractmethod
    def _info(self):
        pass

    @abstractmethod
    def _info_final(self):
        pass

    mean = np.array([ 0., 0., 0.,
                    2000.,
                    0.,0.,
                    0.,
                    0., 0., -15.8,
                    0., 0.,
                    0., 0., 0.
                    ])
    std= np.array([ 0.15, 0.15, 0.15,
                    2000.,
                    5000., 5000.,
                    100.,
                    120., 120., 20.,
                    1., 1.,
                    30., 30., 30.
                    ])    

    def _get_state(self):
        wind = self.sim.get_wind()
        state = np.array([self.sim.sim['velocities/p-rad_sec'],
                        self.sim.sim['velocities/q-rad_sec'],
                        self.sim.sim['velocities/r-rad_sec'],                                                
                        self.pos[2],
                        self.goal[0] -self.pos[0], self.goal[1] - self.pos[1],
                        self.goal[2],
                        self.speed[0], self.speed[1], self.speed[2],
                        self.goal_orientation[0], self.goal_orientation[1],
                        wind[0], wind[1], wind[2]
                        ])
        if not np.isfinite(state).all():
            logging.error('Infinite number detected in state. Replacing with zero')
            logging.error('State: {}'.format(state))
            state = np.nan_to_num(state, neginf=0, posinf=0)
            self._invalid_state = True
        state = (state -self.mean) / self.std
        #state = self.stateNormalizer.normalize(state.view().reshape((1,15)))
        # state = state.view.reshape((15,))
        if not np.isfinite(state).all():
            logging.error('Infinite number after Normalization!')    
            raise ValueError()
        return state

    def __init__(self, save_trajectory = False, render_before_reset=False):
        super().__init__()
        np.random.seed()
        self.config = Config()        
        #print('PyTorch Version: ', torch.__version__)
        self.sim = Sim(sim_dt = 0.02)
        self.save_trajectory = save_trajectory or render_before_reset
        self.render_before_reset=render_before_reset
        self.m_kg = self.sim.sim['inertia/mass-slugs'] * 14.5939029372
        self.g_fps2 = self.sim.sim['accelerations/gravity-ft_sec2']
        self.state: SimState = SimState()
        self.initial_state = SimState()
        self.initial_state.props = self.config.initial_props
        self.plot_fig: plt.Figure = None
        self.trajectory=[]
        self.flightRenderer3D = None
        self.episode_rendered = False
        self.episode = 0
        self.start_date = date.today()
        self.rendered_episode = None
        self._invalid_state = False

        #self.stateNormalizer = Normalizer(self.env_name + '_normalizer', auto_sample=True)

    def _update(self, sim:Sim ):
        self.pos = sim.pos + self.pos_offset
        self.speed = sim.get_speed_earth()
   
    def step(self, action)->Tuple[object, float, bool, dict]: # ->observation, reward, done, info        
        while True:
            self.sim.run()
            self._update(self.sim)
            if self.timer_pid.check_reset(self.sim.time):
                psi = np.nan_to_num(self.sim.sim['attitude/psi-rad'], neginf=0, posinf=0)
                roll = np.nan_to_num(self.sim.sim['attitude/roll-rad'], neginf=0, posinf=0)
                pitch = np.nan_to_num(self.sim.sim['attitude/pitch-rad'], neginf=0, posinf=0)
                heading_target = angle_between(np.array([0,1]), action[0:2])
                #pitch_target = vector_pitch(action)
                pitch_target = 0
                roll_target = self.pid_heading(psi, heading_target)        
                #roll_target=self.pid_heading(self.sim.sim['flight-path/psi-gt-rad'], heading_target)      
                self.sim.sim['fcs/aileron-cmd-norm'] = self.pid_roll(roll, roll_target)
                #self.sim.sim['fcs/elevator-cmd-norm'] = self.pid_pitch(self.sim.sim['flight-path/gamma-rad'], pitch_target)
                self.sim.sim['fcs/elevator-cmd-norm'] = self.pid_pitch(pitch, pitch_target)                  
            #if timer_pid_slip.check_reset(self.sim.time):
            #    self.sim.sim['fcs/rudder-cmd-norm'] = self.pid_slip(self.sim.sim['velocities/v-fps'], 0)
            done = self._done() or self._invalid_state
            if self.timer_goto.check_reset(self.sim.time):# and not goto_arrived:
                if self.save_trajectory: self.trajectory.append(self.pos)
                self.new_state = self._get_state()
                reward = self._reward()
                if not np.isfinite(reward).all():
                    logging.error('Infinite number detected in reward. Replacing with zero')                    
                    reward =-10
                    done = True 
                info = self._info() # info = {}
                return self.new_state, reward, done, info
            if done:
                if self.save_trajectory: self.trajectory.append(self.pos)
                self.new_state = self._get_state()
                reward = self._reward()
                if not np.isfinite(reward).all() or self._invalid_state:
                    logging.error('Infinite number detected in reward or state. Replacing with zero')                    
                    reward =-10
                    done = True
                info = { **self._info(), **self._info_final()} # info = {}
                if self.render_before_reset: self.rendered_episode = self.render(trajectory = self.trajectory)
                return self.new_state, reward, done, info

    def random_position(self, h_range, radius, x_range, y_range, z_range):
        rx1,rx2 = x_range
        ry1,ry2 = y_range
        rz1,rz2 = z_range
        # sx1,sx2 = self.config.x_search_range
        # sy1,sy2 = self.config.y_search_range
        i = 0
        while True:
            i+=1
            if i>100000:
                logging.error('Cannot find random position after 100.000 tries! ')
                raise RuntimeError('Cannot find random position after 100.000 tries! ')
            x = np.random.uniform(rx1, rx2)
            y = np.random.uniform(ry1, ry2)
            dmin, dmax = h_range
            dx, dy  = np.random.uniform(.1, 1.,2)*np.random.choice([-90, 90], 2)
            while rx1<=x<=rx2 and ry1<=y<=ry2:
                h = self.terrain.max_altitude(x, y, radius)
                if h+dmin <= rz2 and rz1 <= h+dmax:
                    z = np.random.uniform(max(h + dmin,rz1), min(h+dmax, rz2))
                    return np.array([x,y,z])
                x += dx
                y += dy    
        
    def reset(self) -> object: #->observation
        self._invalid_state = False
        if self.episode_rendered: 
            self.save_plot()
            self.episode_rendered = False
        (mx1, mx2), (my1,my2) = self.config.map_start_range
        self.terrain.map_offset = [np.random.randint(mx1, mx2), np.random.randint(my1, my2)]
        self.terrain.define_map_for_plotting(self.config.render_range[0], self.config.render_range[1])              
        self.start = self.random_position(self.config.start_ground_distance, self.config.ground_distance_radius,
                                          self.config.x_range_start, self.config.y_range_start, self.config.z_range_start)
        # Startpunkt ins Zentrum des Kartenausschnitts setzen
        self.terrain.map_offset[0] += int(self.start[0] / self.terrain.resolution)
        self.terrain.map_offset[1] += int(self.start[1] / self.terrain.resolution)
        self.terrain.define_map_for_plotting(self.config.render_range[0], self.config.render_range[1])
        self.start[0] = self.start[1] = 0.
        self.goal = self.random_position(self.config.goal_ground_distance, self.config.ground_distance_radius, 
                                        self.config.x_range_goal, self.config.y_range_goal, self.config.z_range_goal)
        self.goal_orientation = np.random.uniform(.01, 1., 3) * np.random.choice([-1,1],3)
        self.goal_orientation[2] = 0
        self.goal_orientation = self.goal_orientation / np.linalg.norm(self.goal_orientation)
        speed_start = np.random.uniform(self.config.speed_range[0], self.config.speed_range[1])
        psi_start = np.random.uniform(0,360)
        self.runway = Runway(self.goal[0:2], self.goal_orientation[0:2],self.config.runway_dimension)
        self.terrain.set_runway(self.runway,self.config.runway_draw)
        self.pos_offset = self.start.copy()
        self.pos_offset[2] = 0
        self.trajectory=[]   
        self.initial_state.position = self.start
        self._reset_sim_state(self.initial_state, speed_start, psi_start)
        #PID-Regler und Timer
        self.pid_pitch = PID_angle('PID pitch', p=-1.5, i=-0.05, d=0, time=0, angle_max=2*np.math.pi, out_min=-1.0, out_max=1.0, anti_windup=1)
        self.pid_roll = PID_angle( 'PID roll', p=17.6, i=0.01, d=35.2, time=0, angle_max=2*np.math.pi, out_min=-1.0, out_max=1.0, anti_windup=1)
        #self.pid_heading = PID_angle('PID heading', p=0.7, i=-0.00002, d=25, time=0, angle_max=2*np.math.pi, out_min=-.6, out_max=.6, anti_windup=1)
        self.pid_heading = PID_angle('PID heading', p=0.7, i=-0.00002, d=25, time=0, angle_max=2*np.math.pi, out_min=-.5*np.math.pi, out_max=.5*np.math.pi, anti_windup=1)
        #self.pid_height = PID('PID height', p=0.7, i=-0.00002, d=25, time=0, out_min=-.1, out_max=.1, anti_windup=1)
        #self.pid_slip = PID('PID slip', p=0.01, i=0.0, d=0, time=0, out_min=-1.0, out_max=1.0, anti_windup=1) #TODO: pid_slip Tunen
        self.timer_pid = SimTimer(0.04, True)
        self.timer_goto = SimTimer(5.)
        self.timer_pid_slip = SimTimer(0.24, True)
        self.roll_target = 0
        self.pitch_target = 0
        self.heading_target = 0
        self.sim.run()
        self._update(self.sim)
        self.episode +=1
        return self._get_state()
    
    def render_start_goal(self, alpha):
        xs,ys, _ = self.start
        self.ax1.plot(xs,ys,'mo', alpha=alpha)
        xs_arr = [x[0] for x in self.runway.arrow]
        ys_arr = [x[1] for x in self.runway.arrow]
        self.ax1.plot(xs_arr, ys_arr, 'm-', alpha=alpha)
        xs_runway = [x[0] for x in self.runway.rectangle] + [self.runway.rectangle[0][0]]
        ys_runway = [x[1] for x in self.runway.rectangle] + [self.runway.rectangle[0][1]]
        self.ax1.plot(xs_runway, ys_runway, 'm-', alpha=alpha)
        xg1,yg1, _ = self.goal            
        self.ax1.plot(xg1,yg1,'m.', alpha=alpha)
        self.ax1.plot([xs,xg1],[ys,yg1],'b-', alpha= .5)         

    def render(self, mode='human', trajectory = None):        
        if self.plot_fig is None:
            cm = 1/2.54  # centimeters in inches
            #self.plot_fig, (self.ax2, self.ax1, self.ax3) = plt.subplots(1,3, gridspec_kw={'width_ratios': (0.5,40,0.5)}, figsize=(20*cm, 16*cm), dpi=80)
            self.plot_fig, self.ax1 = plt.subplots(1,1, figsize=(20*cm, 13*cm), dpi=80)
            #plt.subplots_adjust(left=.01, right=.99, bottom=0.01, top=0.99)
            divider = make_axes_locatable(self.ax1)
            self.ax2 = divider.append_axes("left", size="5%", pad=1.0)
            self.ax3 = divider.append_axes("right", size="5%", pad=0.75)
            #self.ax2.plot([1],[2], cmap='gist_earth', vmin=-1000, vmax = 4000)
            plt.ion()
            plt.show()
        if not self.episode_rendered:            
            self.ax1.clear()
            self.ax2.clear()
            self.ax3.clear()
            plot_dist = np.linalg.norm(self.start[0:2]-self.goal[0:2]) * 1.5
            #(x1,x2), (y1,y2) = self.config.render_range
            (x1,x2), (y1,y2) = (-plot_dist, plot_dist), (-plot_dist, plot_dist)
            res = self.terrain.resolution
            img = self.terrain.get_map((x1,y1), (x2,y2))
            from scipy import ndimage
            img = ndimage.rotate(img, 90)
            im = self.ax1.imshow(img, cmap='gist_earth', vmin=-1000, vmax = 4000, origin='upper', extent=(x1-res/2,x2-res/2,y1-res/2,y2-res/2))
            self.ax1.set_ylabel("Distanz zum Start (Nord-Süd) [m]")
            self.ax1.yaxis.set_label_position('right')
            self.ax1.set_xlabel("Distanz zum Start (West-Ost) [m]")
            self.ax1.xaxis.set_label_position('top')
            cmap = mpl.cm.get_cmap('gist_earth')
            cmap_norm = mpl.colors.Normalize(vmin=-1000, vmax = 4000)
            cb = mpl.colorbar.ColorbarBase(self.ax3, cmap = cmap, norm = cmap_norm, orientation = 'vertical')            
            # cb = plt.colorbar(im,cax = self.ax3)
            self.ax3.yaxis.set_ticks_position('right')
            self.ax3.yaxis.set_label_position('right')
            self.ax3.yaxis.set_ticks([-1000,0,4000])
            cb.set_label("Terrainhöhe [m]", labelpad=-20)
            z_max = self.start[2]
            z_min = self.goal[2]        
            self.cmap = mpl.cm.get_cmap('autumn')
            self.cmap_norm = mpl.colors.Normalize(vmin=z_min, vmax = z_max)
            cb2 = mpl.colorbar.ColorbarBase(self.ax2, cmap = self.cmap, norm = self.cmap_norm, orientation = 'vertical')
            self.ax2.yaxis.set_ticks_position('left')
            self.ax2.yaxis.set_label_position('left')
            self.ax2.yaxis.set_ticks([z_min,z_max])
            cb2.set_label("Flughöhe [m]", labelpad=-20)
            self.plot_fig.tight_layout(pad=1.05)   
            # cb2 = mpl.colorbar.ColorbarBase(self.ax2, cmap=, norm=normalize, orientation='vertical')
            self.render_start_goal(1)
            self.plot_oldxy = self.pos
            self.episode_rendered = True
            self.render_time = time.time()-1            
        x1, y1, z1 = self.plot_oldxy
        x2, y2, z2 = self.pos               
        if trajectory is None:
            self.ax1.plot([x1, x2],[y1, y2], '-', c=self.cmap(self.cmap_norm(z2)) )
        else:
            self.ax1.plot([p[0] for p in trajectory],[p[1] for p in trajectory], '-', c=self.cmap(self.cmap_norm(z2)) )
        if time.time()-self.render_time>0.05:
            plt.gcf().canvas.draw_idle()
            plt.gcf().canvas.start_event_loop(0.0001)
            self.render_time = time.time()
        self.plot_oldxy = self.pos
        if mode=="rgb_array":
            self.plot_fig.canvas.draw()
            image_from_plot = np.frombuffer(self.plot_fig.canvas.tostring_rgb(), dtype=np.uint8)
            image_from_plot = image_from_plot.reshape(self.plot_fig.canvas.get_width_height()[::-1] + (3,))
            return image_from_plot
        return self.plot_fig




    def save_plot(self):
        return
        # save plot in enjopy2.py
        self.render_time = time.time()-1
        self.render()          
        dist_target = np.linalg.norm(self.goal[0:2]-self.pos[0:2])
        delta_angle = abs(angle_between(self.goal_orientation[0:2], self.speed[0:2]))
        delta_angle = np.degrees(delta_angle)
        self.plot_fig = plt.figure('render 2D', figsize=(10, 10), dpi=80)            
        plt.title('Final state: distance to goal={:.2f} m; delta approach angle={:.2f}°; reward={:.2f}'.format(dist_target, delta_angle, self._reward()))
        self.render_start_goal(0.5)
        filename =os.path.join(self.config.logdir,self.env_name,'render','{}_{}_render_episode {}.png'.format(
                                            self.env_name, self.start_date, self.episode))
        filename = ensure_newfile(filename)
        plt.savefig(filename)

    def render_episode_3D(self):        
        if self.flightRenderer3D is None:
            from mayavi import mlab
            from pyface.api import GUI
            self.mlab = mlab 
            self.gui = GUI
            self.flightRenderer3D = plotting.Plotting(None, None, None, self.terrain, self.mlab)
            self.gui.process_events()
            self.save_trajectory = True
        self.flightRenderer3D.plot_map(self.terrain)            
        self.flightRenderer3D.plot_start(self.start)
        #print('Start Position={} goal Position={}'.format(self.start, self.goal))
        self.flightRenderer3D.plot_goal(self.goal, 500)
        self.flightRenderer3D.plot_path(self.trajectory, radius=10)
        self.gui.process_events()

    plot_episode_2d = None

    def render_episode_2D(self):
        pass
        # if self.plot_episode_2d is None:
        #     self.plot_episode_2d = plt.figure('episode 2D')
        #     img = self.terrain.map_window(self.pos[0], self.pos[1], self.OBS_WIDTH, self.OBS_HEIGHT).copy()            
        #     from scipy import ndimage
        #     img = ndimage.rotate(img, 90)
        #     plt.clf()
        #     plt.imshow(img, cmap='gist_earth', vmin=-1000, vmax = 4000, origin='lower', extent=(0,1000,0,1000))
        #     plt.ion()
        #     plt.show()
        #     self.save_trajectory = True
        # plt.figure(self.plot_episode_2d.number)
        # plt.plot()


    def _reset_sim_state(self, state: SimState, speed, psi, engine_on: bool = False):
        state.position = state.position
        state.props['ic/h-sl-ft'] = state.position[2]/0.3048
        state.props['ic/u-fps'] = speed
        state.props['ic/psi-true-rad'] = psi
        self.sim.reinitialise({**self.config.initial_props, **state.props})
        if engine_on:
            self.sim.start_engines()
            self.sim.set_throttle_mixture_controls(0.8, 0.8)
        #logging.debug("load: ",self.jsbSim.pos[2], state.position[2], state.props['ic/h-sl-ft']*0.3048, self.jsbSim['position/h-sl-ft']*0.3048)
        
    def _get_energy(self):
        speed_fps = np.linalg.norm(self.speed)
        e_kin = 1/2 * self.m_kg * speed_fps**2
        h_ft = self.pos[2]    
        e_pot = self.m_kg * self.g_fps2 * h_ft
        return e_pot+e_kin #in kg*ft^2/sec^2
