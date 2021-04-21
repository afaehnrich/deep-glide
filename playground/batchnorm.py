import torch
from torch import nn
import numpy as np

import matplotlib.pyplot as plt


a = np.array([[2,3,5], [4,6,7], [1,5,7]])
a = np.array([1,2,3,4])
b = np.kron(a, np.ones((2,2)))
c= np.ones((8,8))

print(a)
print(a.shape)
print(b)
print(b.shape)
print(np.concatenate([c,b]))