import numpy as np
from deep_glide.utils import angle_between

 

class  DirectToFix():
    def __init__ (self, position: np.array, target: np.array):
        self._target = position
        self.new_target(target)

    def heading_target(self, position: np.array) -> tuple((float, bool)):
        self.on_track_distance = np.dot(self._direction, self._target -  position)
        if self.on_track_distance <=0: 
            return self._target_heading, True
        self._target_heading = angle_between(np.array([0,1]),self._target -  position)        
        return self._target_heading, False
    
    def new_target(self, target: np.array, position: np.array = None):
        if position == None:
            self._start = self._target
        else: 
            self._start = position
        self._target = target
        self._direction = (target - self._start)/np.linalg.norm(target - self._start)
        self.on_track_distance = np.dot(self._direction, self._target -  self._start)   


class TrackToFix():
    #nach "Handbook of unmanned aerial vehicles, chapter 18"

    Tau_star = 10. # Definition des Proportionalitätsfaktors für L2
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

    def roll_target(self, position: np.array, velocity_ground: np.array) -> tuple((float, bool)):
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
        self.D_wp1 = np.dot(self._direction, self._target-position) # eq. (18.47)
        D_a = self.D_wp1 - D_dt_star_min # eq. (18.48)
        P_a = -self._direction*max(0, D_a) + self._target # eq. (18.49)
        L2 = P_a - position
        sineta = np.cross(velocity_ground, L2) / (np.linalg.norm(velocity_ground) * np.linalg.norm(L2)) # eq. (18.35)
        a_cmd = -2 * np.linalg.norm(velocity_ground)/ self.Tau_star * sineta # eq. (18.43)
        if sineta > np.math.pi/2:
            a_cmd = 9,81* np.math.tan(0.7)
            print("maxroll")
        roll_cmd = np.math.atan(a_cmd/ 9.81)
        if self.D_wp1 <= self._switching_distance:
            return 0, True
        else:
            return roll_cmd, False


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
            return 0
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
        pitch_target = angle_between(P_a-position, np.array([1,0]))
        return pitch_target

class TrackToFix3D():
    
    def __init__(self, target: np.array, position: np.array, switching_distance: float = None):
        self.ttf = TrackToFix(target[0:2], position[0:2], switching_distance)

        P0_ =np.array([0, position[2]])
        P1_ =np.array([np.linalg.norm(target[0:2]-position[0:2]), target[2]])
        self.ttfh = TrackToFixHeight(P1_, P0_, switching_distance)
        
    def new_target(self, target: np.array, position: np.array, switching_distance: float = None):
        if position is None:
            self.ttf.new_target(target[0:2], switching_distance= switching_distance)
        else:
            self.ttf.new_target(target[0:2], position[0:2], switching_distance)
        P0_ =np.array([0, position[2]])
        P1_ =np.array([np.linalg.norm(target[0:2]-position[0:2]), target[2]])
        self.ttfh.new_target(P1_, P0_, switching_distance)

    def get_commands(self, position: np.array, velocity_ground: np.array) -> tuple((float, float, bool)):
        roll_cmd, arrived = self.ttf.roll_target(position[0:2], velocity_ground[0:2])
        Puav_ = np.array([self.ttfh._target[0]-self.ttf.D_wp1, position[2]])
        speed_ = np.array([np.linalg.norm(velocity_ground[0:2]), velocity_ground[2]])
        pitch_cmd = self.ttfh.pitch_target(Puav_, speed_)
        return roll_cmd, pitch_cmd, arrived
