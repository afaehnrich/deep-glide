import torch
from torch import nn
import numpy as np

from numpy import convolve
import matplotlib.pyplot as plt
from deep_glide.jsbgym_new.sim_handler_rl import JSBSimEnv_v0
from  deep_glide.jsbgym_new.sim import Sim
 

dfnum = 3 # between group deg of freedom
dfden = 20 # within groups degrees of freedom
nonc = 3.0

class Normalizer:
    mean = 1.
    variance = 1.
    squaresum = 0.
    n = 0.

    def add_sample(self,x):
        self.mean = self.mean*self.n
        self.variance = self.variance * self.n
        self.n +=1.
        self.mean = (self.mean +x)/self.n
        self.squaresum += x**2
        self.variance = 1/(self.n)*self.squaresum - self.mean**2
        self.std = np.sqrt(self.variance)

    def normalize(self, x, add_sample=True):
        if add_sample: self.add_sample(x)
        return (x - self.mean) / np.sqrt(self.std**2+0.00001)


def mittelw():
    mw = Normalizer()
    
    for i in range(0,20000):
        v = np.random.noncentral_f(dfnum, dfden, nonc, 4)
        mw.add_val(v)
    return mw.mean, mw.std

def mwstd():    
    vs = np.random.noncentral_f(dfnum, dfden, nonc, (20000,4))
    for i in range(1,len(vs)):
        m,s = np.mean(vs[0:i],axis=0), np.std(vs[0:i], axis=0)
    return m, s

exit()
np.set_printoptions(precision=2, suppress=True)
nc_vals = np.random.noncentral_f(dfnum, dfden, nonc, 4)
firstvals=np.random.noncentral_f(dfnum, dfden, nonc, (10000,4))*1000.
print (firstvals)
import timeit
# print(timeit.timeit(lambda: print(mittelw()), number=1))
# print(timeit.timeit(lambda: print(mwstd()), number=1))

m =nn.BatchNorm1d(4).float()
x=m(torch.from_numpy( firstvals).float())
print (firstvals[-1])
print(x[-1])
mw = Normalizer()
for v in firstvals:
    mw.add_sample(v)
print(mw.normalize(firstvals[-1]))
print (mw.mean, mw.std)


