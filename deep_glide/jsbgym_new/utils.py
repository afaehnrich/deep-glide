import numpy as np


def limit_angle( angle, max_angle):
    #limitiert den Winkel auf einen +/-halben Kreis, z.B. auf max. -180°..180°
    half = max_angle / 2
    return (angle + half ) % max_angle - half

class Normalizer:
    mean = 1.
    variance = 1.
    squaresum = 1.
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
