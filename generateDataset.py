# Make a bunch of dataset types
import numpy as np
import math

class datasets:
    def __init__(self):
        self.x = 1

    def binClassifier(self, n, func, inz=0, fin=1):
        x = np.random.rand(n, 2) * (fin - inz) + inz
        y = np.zeros((n, 2))
        for pos, coords in enumerate(x):
            y[pos, 0] = func(((coords[0]-inz)/fin, (coords[1]-inz)/fin))
            y[pos, 1] = 1-y[pos, 0]
        return x, y

    def parabola(self):
        pass