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

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def ensure_newfile(file_path):    
    ensure_dir(file_path)
    if not os.path.exists(file_path): return file_path    
    directory, fn_new = os.path.split(file_path)
    filename, extension = os.path.splitext(fn_new)
    i=0
    while os.path.exists(os.path.join(directory,fn_new)):
        i +=1
        fn_new = '{}_{}{}'.format(filename,i,extension)
    return os.path.join(directory,fn_new)


class Normalizer:
    save_count = 0
    n_samples = 0
    count =1
    mean = 1
    M2 = 1
    auto_sample=False

    def __init__(self, simName, auto_sample= False, save_interval=50000):
        self.simName = simName
        self.save_interval = save_interval
        self.auto_sample = auto_sample

    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    # For a new value newValue, compute the new count, new mean, the new M2.
    # mean accumulates the mean of the entire dataset
    # M2 aggregates the squared distance from the mean
    # count aggregates the number of samples seen so far
    def add_batch(self, newValues: np.array):
        if not np.isfinite(newValues).all(): return
        valCount = newValues.shape[0]
        valSum = newValues.sum(axis =0)
        self.count += valCount
        delta = valSum - self.mean * valCount
        delta = np.nan_to_num(delta, neginf=0, posinf=0)
        self.mean += delta / self.count
        self.mean = np.nan_to_num(self.mean, neginf=0, posinf=0)
        delta2 = valSum  - self.mean *valCount
        delta2 = np.nan_to_num(delta2, neginf=0, posinf=0)
        self.M2 += delta * delta2
        self.M2 = np.nan_to_num(self.M2, neginf=0, posinf=0)

    def save_state(self):
        with open('{}_normalizer.txt'.format(self.simName), 'a') as file: 
            variance = self.M2 / self.count
            std = np.sqrt(variance)
            file.write('n_samples={}:\r\n  count={}\r\n  mean={}\r\n  M2={}\r\n  variance={}\r\n  std={}\r\n\r\n'.format(self.n_samples, self.count, self.mean,
                     self.M2, variance, std))        

    def normalize(self, x: np.array) -> np.array:
        if self.auto_sample: 
            self.add_batch(x)
            self.save_count += x.shape[0]
            self.n_samples += x.shape[0]
            if self.save_count >= self.save_interval:
                self.save_state()            
                self.save_count -= self.save_interval
        variance = self.M2 / self.count
        std = np.sqrt(variance)
        return (x - self.mean) / np.sqrt(std**2+0.00001)

class Normalizer2D(Normalizer):
    # All pixels are threated equal
    def add_batch(self, newValues: np.array):
        xdim, ydim, zdim = newValues.shape            
        super().add_batch(newValues.view().reshape(xdim*ydim*zdim,1))


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