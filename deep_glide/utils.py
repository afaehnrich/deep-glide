import numpy as np
import logging
import os
from array import array

def angle_between(v1, v2):
    #nach https://stackoverflow.com/a/13849249/11041146
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    cross = np.cross(v2,v1)
    sign = np.sign(cross)
    if sign == 0: sign = 1
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    if np.isnan(angle): return 0
    return angle*sign

def angle_plane_line(v1_plane:np.array , v2_plane: np.array, v_line: np.array)->float:
    n = np.cross(v1_plane, v2_plane)
    cosalpha = np.linalg.norm(np.cross(n, v_line)) / (np.linalg.norm(n) * np.linalg.norm(v_line))
    return np.math.acos(cosalpha)
    
def vector_pitch(v: np.array)->float:
    n = np.array([0.,0.,1.]) # Normalenvektor der XY-Ebene
    cosalpha = np.linalg.norm(np.cross(n, v)) / (np.linalg.norm(n) * np.linalg.norm(v))
    return np.math.acos(cosalpha)* np.sign(v[2]) # ist Z negativ, ist der Pitch kleiner Null
   

def limit_angle( angle, max_angle):
    #limitiert den Winkel auf einen +/-halben Kreis, z.B. auf max. -180°..180°
    half = max_angle / 2
    return (angle + half ) % max_angle - half

class Normalizer:
    mean = 1.
    variance = 1.
    squaresum = 1.
    std = 1.
    n = 1.
    epsion = 0.0001

    def add_sample(self,x):
        _mean = self.mean
        _std = self.std
        _var = self.variance
        _squaresum = self.squaresum
        self.mean = self.mean*self.n
        self.variance = self.variance * self.n
        self.n +=1.
        self.mean = (self.mean +x)/self.n
        self.mean = np.nan_to_num(self.mean, neginf=0, posinf=0)          
        self.squaresum += x**2
        self.squaresum = np.nan_to_num(self.squaresum, neginf=0, posinf=0)
        self.variance = 1/(self.n)*self.squaresum - self.mean**2        
        self.variance = np.nan_to_num(self.variance, neginf=0, posinf=0)
        self.std = np.sqrt(self.variance)
        self.std = np.nan_to_num(self.std, neginf=0, posinf=0)        

    def normalize(self, x, add_sample=True):
        if add_sample: self.add_sample(x)
        return (x - self.mean) / np.sqrt(self.std**2+0.00001)

def elevation_asc2hgt(filename_asc, filename_hgt):
    path = os.path.dirname(os.path.realpath(__file__))
    fname = os.path.join(path,filename_asc)
    fname_save = os.path.join(path,filename_hgt)
    data =[]
    from datetime import datetime
    t1 = datetime.now()
    import csv
    with open(fname, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')#, quoting=csv.QUOTE_NONNUMERIC)
        for _ in range(0,6): next(reader)
        data = list(reader)
    data=[row[:-1] for row in data]
    data = [item for sublist in data for item in sublist]
    data = list(map(int, data))
    t2=datetime.now()
    print(t2-t1)
    format = 'h'    
    data2 = array(format)
    data2.fromlist(data)
    data2.byteswap()
    f = open(fname_save, 'wb')
    data2.tofile(f)