from typing import Counter
import torch
import gym
import torch
from torch import nn
import numpy as np
from torch.functional import norm
import psutil

        # Check that the replay buffer can fit into the memory
mem_available = psutil.virtual_memory().available
print(mem_available/1024/1024/1024)
exit()

class Normalizer_old:
    mean = 1.
    variance = 1.
    squaresum = 1.
    std = 1.
    n = 1.
    epsion = 0.0001
    simName: str
    save_data: int

    def __init__(self, simName, save_data=500000):
        self.simName = simName
        self.save_data = save_data

    def add_sample(self,x):
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
        if self.n % self.save_data == 0:
            with open('{}_normalizer.txt'.format(self.simName), 'a') as file: 
                file.write('n_step={}\r\nSquaresum={}\r\nMean={}\r\nVar={}\r\nStd={}\r\n\r\n'.format(self.n, self.squaresum, self.mean, self.variance, self.std))        

    def normalize(self, x, add_sample=True):
        if add_sample: self.add_sample(x)
        return (x - self.mean) / np.sqrt(self.std**2+0.00001)

class Normalizer:
    count =1
    mean = 1
    M2 = 1

    def __init__(self, simName, save_data=50000):
        self.simName = simName
        self.save_data = save_data

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
            file.write('count={}\r\nmean={}\r\nM2={}\r\n\r\n'.format(self.count, self.mean, self.M2))        

    def normalize(self, x, add_sample=True):
        if add_sample: 
            self.add_batch(x)
            if self.n % self.save_data == 0: self.save_state()            
        variance = self.M2 / self.count
        std = np.sqrt(variance)
        return (x - self.mean) / np.sqrt(std**2+0.00001)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


dfnum = 3 # between group deg of freedom
dfden = 20 # within groups degrees of freedom
nonc = 3.0
nc_vals = np.random.noncentral_f(dfnum, dfden, nonc, size=(1000,2))
#nc_vals = nc_vals * np.array([1,100])
#print(nc_vals)
#exit()
normalizer = Normalizer_old('xyz')
normalizer2 = Normalizer('xyz')
normalizer3 = Normalizer('xyz')
normalizer4 = Normalizer('xyz')
from torch import nn
bn = nn.BatchNorm1d(2)
#print (nc_vals)
import time
t1 = time.time()
for v in nc_vals:
    normalizer.add_sample(v)
print(time.time()-t1)
t1 = time.time()
for v in nc_vals:
    normalizer2.add_batch(v.view().reshape(1,2))
print(time.time()-t1)
t1=time.time()
normalizer4.add_batch(nc_vals)
print(time.time()-t1)
from itertools import islice
iterator = islice(nc_vals, 10)
for v in chunks(nc_vals,10):
    normalizer3.add_batch(v)
print()
print( normalizer.normalize(nc_vals[0], add_sample=False), normalizer.mean, normalizer.n)
print( normalizer2.normalize(nc_vals[0], add_sample=False), normalizer2.mean, normalizer2.count, normalizer2.M2)
print( normalizer3.normalize(nc_vals[0], add_sample=False), normalizer3.mean, normalizer3.count, normalizer3.M2)
print( normalizer3.normalize(nc_vals[0], add_sample=False), normalizer4.mean, normalizer4.count, normalizer4.M2)
print( normalizer3.normalize(nc_vals[0:2], add_sample=False), normalizer4.mean, normalizer4.count, normalizer4.M2)
print( bn(torch.from_numpy(nc_vals).float())[0])
print(nc_vals.mean())


class Normalizer2D(Normalizer):

    def add_batch(self, newValues: np.array):
        xdim, ydim, zdim = newValues.shape            
        super().add_batch(newValues.view().reshape(xdim*ydim*zdim,1))


nc_vals = np.random.noncentral_f(dfnum, dfden, nonc, size=(10,4,3))
nc_vals = nc_vals *100
nc_vals[2,2,2]= np.math.inf
print (nc_vals)
print()
normalizer2d = Normalizer2D('xyz')
normalizer2d.add_batch(nc_vals)
print('--------------------------')
print( normalizer2d.normalize(nc_vals[0:2], add_sample=False))
print('--------------------------')
print(normalizer2d.mean, normalizer2d.count, normalizer2d.M2)
