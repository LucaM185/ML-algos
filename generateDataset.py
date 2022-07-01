# Make a bunch of dataset types
import numpy as np
import math

class datasets:
    def __init__(self):
        self.x = 1

    def dotChart(self, n, inz, fin, func):
        x = np.random.rand(n, 2) * (fin - inz) + inz
        y = np.zeros((n, 2))
        for pos, coords in enumerate(x):
            y[pos, 0] = func(((coords[0]-inz)/fin, (coords[1]-inz)/fin))
            y[pos, 1] = 1-y[pos, 0]
        return x, y