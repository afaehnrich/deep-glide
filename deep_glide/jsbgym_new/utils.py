import numpy as np
import logging
import os
from array import array


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