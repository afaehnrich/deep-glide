import numpy as np
from mayavi import mlab
from typing import Tuple

figure = mlab.figure(size=(800,800))

def plot_dir(p:np.array, color:Tuple):
    return mlab.plot3d([0, p[0]], [0, p[1]], [0, p[2]], tube_radius=10, color=color, opacity =.4)



def angle_plane_line(v1_plane:np.array , v2_plane: np.array, v_line: np.array)->float:
    n = np.cross(v1_plane, v2_plane)
    cosalpha = np.linalg.norm(np.cross(n, v_line)) / (np.linalg.norm(n) * np.linalg.norm(v_line))
    alpha = np.math.acos(cosalpha)
    plot_dir(v1_plane*100, (0,0,1))
    plot_dir(v2_plane*100, (0,0,1))
    plot_dir(v_line*100, (1,0,0))
    plot_dir(n*100, (0,1,0))
    #if np.sign()
    return alpha

def vector_pitch(v: np.array)->float:
    n = np.array([0.,0.,1.]) # Normalenvektor der XY-Ebene
    cosalpha = np.linalg.norm(np.cross(n, v)) / (np.linalg.norm(n) * np.linalg.norm(v))
    return np.math.acos(cosalpha)* np.sign(v[2]) # ist Z negativ, ist der Pitch kleiner Null

v1_plane = np.array([1, 0, 0])
v2_plane = np.array([0, 1, 0])
v_line = np.array([0,-10,-1])

a = angle_plane_line(v1_plane,v2_plane, v_line )
print(a, np.math.degrees(a))
print (np.cross(v1_plane, v2_plane))

input()

