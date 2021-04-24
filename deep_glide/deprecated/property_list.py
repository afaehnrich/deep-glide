import collections
from deep_glide.envs.abstractSimHandler import AbstractJSBSimEnv as SH
import math
import numpy as np
from deep_glide.properties import BoundedProperty2, Property2
from deep_glide.sim import Sim


class Properties2:
    #s = Sim(0.02)

    #a = s.__getitem__('position/h-sl-ft')

    def __init__(self, simHandler):

        #General properties
        self.infinity = BoundedProperty2(-math.inf, math.inf, None, None)
        #custom properties
        self.custom_dir_x = BoundedProperty2(-1., 1, None, None)
        self.custom_dir_y = BoundedProperty2(-1., 1, None, None)
        self.custom_dir_z = BoundedProperty2(-1., 1, None, None)
        self.position_h_sl_ft = BoundedProperty2(-1400., 85000., SH.sim.pos, SH.sim.__getitem__('position/h-sl-ft'))
