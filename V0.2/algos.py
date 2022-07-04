import numpy as np
from numpy import sin, cos
import cv2
from math import pi
import random
from numba import njit
import time

np.set_printoptions(suppress=True)


class linearRegression:
    def __init__(self, nSamples=200, minLimit=0, maxLimit=1):
        self.n = nSamples
        self.nMax = maxLimit
        self.nMin = minLimit

    def makedataset(self, function, randomness):
        self.x = np.arange(self.nMin, self.nMax, (self.nMax-self.nMin)/self.n)
        self.y = np.zeros(self.x.shape[0])
        todelete = []
        for i, n in enumerate(self.x):
            ypred = function(n) + (random.random()-0.5) * randomness * 2
            if ypred < self.nMin or ypred > self.nMax:
                todelete.append(i)
            self.y[i] = ypred
        for d in reversed(todelete):
            self.x = np.delete(self.x, d)
            self.y = np.delete(self.y, d)

        # As a convention: ALWAYS use -y when displaying info on cv2 window
    def chart(self, res=200, dotSize=5):
        self.img = np.zeros((res, res, 3), dtype=np.uint8)
        for i in range(self.x.shape[0]):
            cv2.circle(self.img, (int(res*self.x[i]), res-int(res*self.y[i])), dotSize, (255, 0, 0), -1)

    def drawPredictions(self):
        res = self.img.shape[0]
        for x in np.arange(0, 1, 1/400):
            y = x*self.best[0] + self.best[1]  # y = mx + q
            cv2.circle(self.img, (int(x*res), res-int(y*res)), 1, (0, 0, 255), -1)


    def show(self, res=800):
        cv2.imshow("w1", cv2.resize(self.img, (800, 800)))
        cv2.waitKey(10)

    def showWait(self, res=800):
        cv2.imshow("w1", cv2.resize(self.img, (800, 800)))
        return cv2.waitKey(0)


    def train(self, mSamples, qSamples):
        t0 = time.time()
        self.msamples, self.qsamples = mSamples, qSamples

        # Calculating loss for many parameters of: y = mx + q
        # Making a set of angular coeffs
        mSamples += 1
        mTarget = 0
        mDistribution = pi/2
        mSteps = mDistribution * 2 / mSamples
        aMin = mTarget-mDistribution+mSteps
        aMax = mTarget+mDistribution
        ang = np.arange(aMin, aMax, mSteps)
        angularCoefficients = sin(ang) / cos(ang)

        # Making a set of q
        '''
        qTarget = 0.5
        qDistribution = 1
        qSteps = qDistribution * 2 / qSamples
        qMin = qTarget-qDistribution+qSteps
        qMax = qTarget+qDistribution
        qs = np.arange(qMin, qMax, qSteps)
        '''

        qTarget = 0.5
        qDistribution = 4
        qs = np.arange(-1, 1, 1/qSamples)
        qs = qs*abs(qs)
        qs += qTarget
        qs *= qDistribution
        print(qs)




        ys = self.y
        xs = self.x

        self.best = fastTraining(ys, xs, angularCoefficients, qs)
        return time.time()-t0



@njit()
def fastTraining(ys, xs, ms, qs):
    best = None
    minLoss = 1e15
    for m in ms:
        for q in qs:
            loss = 0
            for i, x in enumerate(xs):
                pred = x * m + q
                loss += (pred - ys[i]) ** 2
            if loss < minLoss:
                minLoss = loss
                best = [m, q]
    return best
