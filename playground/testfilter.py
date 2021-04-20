from operator import pos
from matplotlib import legend
from deep_glide.jsbgym_new.sim import Sim
import matplotlib.pyplot as plt
from matplotlib import gridspec
from datetime import datetime
from deep_glide.jsbgym_new.pid import PID_angle, PID
from deep_glide.jsbgym_new.guidance import TrackToFix, DirectToFix
sim = Sim(sim_dt = 0.02)
import numpy as np
import random
from typing import Tuple, Dict, List

random.seed()
np.random.seed()



def is_higher(node_start, node_end):
    return node_start.np[2] > node_end.np[2]

def nearest_neighbor(node_list, n):
    # Nur Nodes einbeziehen, die höher liegen
    node_list_filtered = filter(lambda nd: nd.np[2] > n.np[2], node_list)

    return nn

node_list=[]
for i in range(0,20):
    node_list.append(np.random.uniform(-100,100,3))
for n in node_list:
    print(n)
print()
print()
print()


n=(10,10,10)
node_list_filtered = filter(lambda nd: nd[2] > n[2], node_list)
for nd in node_list_filtered:
    print(nd)
print('---------------')
for nd in node_list_filtered:
    print(nd, nd[2], n[2])