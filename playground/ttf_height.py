from operator import pos
from matplotlib import legend
from deep_glide.jsbgym_new.sim import Sim
import matplotlib.pyplot as plt
from matplotlib import gridspec
from datetime import datetime
from deep_glide.jsbgym_new.pid import PID_angle, PID
from deep_glide.jsbgym_new.guidance import TrackToFix, DirectToFix, angle_between
sim = Sim(sim_dt = 0.02)
import numpy as np
import random
from typing import Tuple, Dict, List

random.seed()
np.random.seed()

def plotpoint3d(p, format):
    plt.plot(p[0],p[1],p[2], format)

def plotline3d(p0, p1, format):
    plt.plot([p0[0], p1[0]],[p0[1], p1[1]],[p0[2], p1[2]], format)

def plotpoint(p, format):
    plt.plot(p[0],p[1], format)

def plotline(p0, p1, format):
    plt.plot([p0[0], p1[0]],[p0[1], p1[1]], format)



class TrackToFixHeight():
    #nach "Handbook of unmanned aerial vehicles, chapter 18"

    Tau_star = 2. # Definition des Proportionalitätsfaktors für L2
    intercept_angle_max = 1.
    M_star_DP = 0.002
    _switching_distance: int = 200

    def __init__(self, target: np.array, position: np.array, switching_distance: float = None):
        self.new_target(target, position, switching_distance)
        
    def new_target(self, target: np.array, position: np.array = None, switching_distance: float = None):
        if position is None:
            self._start = self._target
        else: 
            self._start = position
        self._target = target
        self._direction = (target - self._start)
        self.N = np.array([-self._direction[1], self._direction[0]]) # eq. (18.37)
        self.N = 1 / np.linalg.norm(self.N) * self.N # eq. (18.37)

        distance = np.linalg.norm(target - self._start)
        if distance != 0: self._direction = self._direction / distance
        if switching_distance is not None: self._switching_distance = switching_distance

    def pitch_target(self, position: np.array, velocity_ground: np.array) -> tuple((float, bool)):
        if np.linalg.norm( self._start - self._target) == 0 or np.linalg.norm (self._target - position) == 0:
            return 0, True
        e_N = np.dot(self.N,  (position - self._start)) # eq. (18.38)
        # SLUG L2+-control
        len_L2 = self.Tau_star * np.linalg.norm(velocity_ground)
        D_dt = e_N / np.math.tan(self.intercept_angle_max) # eq. (18.44)
        D_dt_min = min (D_dt, len_L2 * self.M_star_DP) # eq. (18.45)              
        if abs(e_N) <= len_L2: # eq. (18.46)
            D_dt_star_min = max(D_dt_min,np.math.sqrt(len_L2**2 - e_N**2))
        else:
            D_dt_star_min = D_dt_min
        D_wp1 = np.dot(self._direction, self._target-position) # eq. (18.47)
        D_a = D_wp1 - D_dt_star_min # eq. (18.48)
        P_a = -self._direction*max(0, D_a) + self._target # eq. (18.49)
        return P_a


P0=np.array([200,0,1000])
P1=np.array([1000,0,1000])
Puav=np.array([300,0,1100])
speed_g = np.array([120,2,10])
speed_ = np.array([np.linalg.norm(speed_g[0:2]), speed_g[2]])
ttf = TrackToFix(P1[0:2], P0[0:2], 0)
_, _ = ttf.roll_target(np.array([Puav[0], Puav[1]]), speed_g[0:2])

direction = (P1-P0)/np.linalg.norm(P1-P0)
E= direction * np.linalg.norm(P0-Puav)
fig = plt.figure(figsize=(14, 7)) 
#plt.clf()
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1]) 
ax0 = plt.subplot(gs[0], projection='3d')
plotpoint3d(P0,'g.')
plotpoint3d(P1,'r.')
plotpoint3d(Puav,'b.')
plotline3d(P0, P1, 'y-')
ax1 = plt.subplot(gs[1])
P0_ =np.array([0, P0[2]])
P1_ =np.array([np.linalg.norm(P1[0:2]-P0[0:2]), P1[2]])
Puav_ = np.array([P1_[0]-ttf.D_wp1, Puav[2]])
#E_=(P0_-P1_)/np.linalg.norm(P0_-P1_)*ttf.D_wp1
#E_ = E_ + P1_

ttfh = TrackToFixHeight(P1_, P0_)
P_a = ttfh.pitch_target(Puav_, speed_)

plotpoint(P0_,'g.')
plotpoint(P1_,'r.')
plotpoint(Puav_,'b.')
#plotpoint(E_,'y.')
plotpoint(P_a,'c.')
plotline(P0_, P1_, 'y-')
plotline(Puav_, Puav_ + speed_, 'r-')
#plotline(Puav_, E_,'b-')
pitch = angle_between(P_a-Puav_, np.array([1,0]))
print(pitch)
plt.show()
