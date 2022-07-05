import numpy
import numpy as np
from numpy import sin, cos
import cv2
from math import pi
import random
from numba import njit
import time

np.set_printoptions(suppress=True)


# As a convention: ALWAYS use -y when displaying info on cv2 window
class display:
    def chart(self, res=200, dotSize=5, size=1, center=False):
        self.img = np.zeros((res, res, 3), dtype=np.uint8)
        for i in range(self.x.shape[0]):
            cv2.circle(self.img, (int(res * self.x[i] / size), res - int(res * self.y[i] / size + res / 2 * center)),
                       dotSize, (255, 0, 0), -1)

    def show(self, res=800, time=0):
        cv2.imshow("w1", cv2.resize(self.img, (res, res)))
        return cv2.waitKey(time)


##########################################################################################
####################           DECISION TREES                    #########################
##########################################################################################

class branch:
    def __init__(self, parameters, depth, maxdepth, range):
        depth+=1
        self.condition = [random.choice(parameters), random.randrange(range[0], range[1])]
        # This is only going to work for one input parameter

    def compute(self, x):
        p, n = self.condition
        activation = p > n
        if activation:
            # other branch with higher range
        else:
            # other branch with lower range



class decisionTrees(display):
    def __init__(self):
        pass

    def makeDataset(self, n=200, function=numpy.sin, minimum=0, maximum=30):
        # Making stuff more accessible
        self.n = n
        self.func = function
        self.minimum = minimum
        self.maximum = maximum

        # Generating x coords of a chart
        self.x = np.arange(minimum, maximum, maximum / n)
        self.y = np.sin(self.x)

    def makeRandomTree(self, depth=0):



##########################################################################################
####################           LINEAR REGRESSION                 #########################
##########################################################################################

class linearRegression(display):
    def __init__(self, nSamples=200, minLimit=0, maxLimit=1):
        self.n = nSamples
        self.nMax = maxLimit
        self.nMin = minLimit
        self.x = np.zeros(0)

    def makedataset(self, function, randomness):
        self.x = np.arange(self.nMin, self.nMax, (self.nMax - self.nMin) / self.n)
        self.y = np.zeros(self.x.shape[0])
        todelete = []
        for i, n in enumerate(self.x):
            ypred = function(n) + (random.random() - 0.5) * randomness * 2
            if ypred < self.nMin or ypred > self.nMax:
                todelete.append(i)
            self.y[i] = ypred
        for d in reversed(todelete):
            self.x = np.delete(self.x, d)
            self.y = np.delete(self.y, d)

    def dataset(self, x, y):
        self.x, self.y = x, y

    def drawPredictions(self, color=(0, 0, 255)):
        res = self.img.shape[0]
        for x in np.arange(0, 1, 1 / 400):
            y = x * self.best[0] + self.best[1]  # y = mx + q
            cv2.circle(self.img, (int(x * res), res - int(y * res)), 1, color, -1)

    def drawLine(self, m, q, color=(0, 0, 255), dotSize=1):
        res = self.img.shape[0]
        for x in np.arange(0, 1, 1 / 400):
            y = m * x + q  # y = mx + q
            if y < res and y > 0:
                cv2.circle(self.img, (int(x * res), res - int(y * res)), dotSize, color, -1)

    def train(self, mSamples, qSamples, EPOCHS=1, mTarget=0, qTarget=None, draw=1):
        self.draw = draw
        self.losses = []
        totalTime = 0
        if qTarget == None:
            qTarget = np.mean(self.y)
        self.best = [mTarget, qTarget]
        for EPOCH in range(EPOCHS):
            time, self.best, loss = self.trainOnce(mSamples, qSamples, iteration=EPOCH, mTarget=self.best[0],
                                                   qTarget=self.best[1])
            totalTime += time
            self.losses.append(round(loss, 4))
            if draw == 1 or EPOCH == EPOCHS - 1:
                self.drawPredictions((0, 0, 32 + 223 * (EPOCH / EPOCHS)))
        return totalTime, self.best, self.losses

    def trainOnce(self, mSamples, qSamples, iteration=0, mTarget=0, qTarget=0.5):
        t0 = time.time()
        self.msamples, self.qsamples = mSamples, qSamples

        # Calculating loss for many parameters of: y = mx + q
        # Making a set of angular coeffs
        aTarget = np.arctan(mTarget)
        mDistribution = pi / 2 / 2 ** iteration
        angs = np.arange(-1, 1, 1 / qSamples)
        angs = aTarget + mDistribution * angs
        ms = np.tan(angs)

        # Making a set of q
        qDistribution = 4 * np.mean(self.y) / 2 ** iteration
        qs = np.arange(-1, 1, 1 / qSamples)
        qs = qs * abs(qs) * qDistribution + qTarget

        ys = self.y
        xs = self.x

        self.best, loss = fastTraining(ys, xs, ms, qs)

        # Show yellow lines
        if iteration == 0 and self.draw == 2:
            for m in ms:
                self.drawLine(m, self.best[1], (0, 30 + 20 * (iteration + 1), 0), dotSize=1)

        return time.time() - t0, self.best, loss

    def Rsquared(self):
        varMean = stdDev(self.y)
        varFit = self.losses[-1]
        return (varMean - varFit) / varMean

    def predict(self, x):
        m, q = self.best
        return x * m + q


def stdDev(y):
    mean = np.mean(y)
    stdDeviation = 0
    for n in y:
        stdDeviation += (n - mean) ** 2
    return stdDeviation


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
    return best, minLoss


##########################################################################################
####################           LOGISTIC REGRESSION                 #######################
##########################################################################################

class logisticRegression(display):
    def __init__(self, nSamples=200, minLimit=0, maxLimit=1):
        self.n = nSamples
        self.nMax = maxLimit
        self.nMin = minLimit
        self.x = np.zeros(0)

    def makedataset(self, function, randomness):
        self.x = np.arange(self.nMin, self.nMax, (self.nMax - self.nMin) / self.n)
        self.y = np.zeros(self.x.shape[0])
        todelete = []
        for i, n in enumerate(self.x):
            ypred = function(n) + (random.random() - 0.5) * randomness * 2
            if ypred < self.nMin or ypred > self.nMax:
                todelete.append(i)
            self.y[i] = ypred
        for d in reversed(todelete):
            self.x = np.delete(self.x, d)
            self.y = np.delete(self.y, d)

    def dataset(self, x, y):
        self.x, self.y = x, y

        # As a convention: ALWAYS use -y when displaying info on cv2 window

    def drawPredictions(self, color=(0, 0, 255)):
        res = self.img.shape[0]
        for x in np.arange(0, 1, 1 / 400):
            y = (x ** 2) * self.best[0] + x * self.best[1] + self.best[2]  # y = ax2 + bx + c
            cv2.circle(self.img, (int(x * res), res - int(y * res)), 1, color, -1)

    def train(self, samples, EPOCHS=1, targets=[0, 0, 0], draw=1):
        self.draw = draw
        self.losses = []
        totalTime = 0
        self.best = targets
        for EPOCH in range(EPOCHS):
            time, self.best, loss = self.trainOnce(samples, iteration=EPOCH, aTarget=self.best[0], bTarget=self.best[1],
                                                   cTarget=self.best[2])
            totalTime += time
            self.losses.append(round(loss, 4))
            if draw == 1 or EPOCH == EPOCHS - 1:
                self.drawPredictions((0, 0, 32 + 223 * (EPOCH / EPOCHS)))
        return totalTime, self.best, self.losses

    def trainOnce(self, samples, iteration=0, aTarget=0, bTarget=0, cTarget=0):
        t0 = time.time()
        self.aSamples, self.bSamples, self.cSamples = samples

        # Making a set of a
        ADistribution = 10 / 2 ** iteration
        As = np.arange(-1, 1, 1 / self.aSamples)
        As = As * abs(As) * ADistribution + aTarget
        # Making a set of b
        BDistribution = 10 / 2 ** iteration
        Bs = np.arange(-1, 1, 1 / self.bSamples)
        Bs = Bs * abs(Bs) * BDistribution + bTarget
        # Making a set of c
        CDistribution = 10 / 2 ** iteration
        Cs = np.arange(-1, 1, 1 / self.cSamples)
        Cs = Cs * abs(Cs) * CDistribution + cTarget

        ys = self.y
        xs = self.x

        self.best, loss = fastTrainingLogistic(ys, xs, As, Bs, Cs)

        return time.time() - t0, self.best, loss

    def Rsquared(self):
        varMean = stdDev(self.y)
        varFit = self.losses[-1]
        return (varMean - varFit) / varMean

    def predict(self, x):
        a, b, c = self.best
        return x ** 2 * a + x * b + c

@njit()
def fastTrainingLogistic(ys, xs, As, Bs, Cs):  # EPOCH * n**5 complexity !!
    best = None
    minLoss = 1e15
    for a in As:
        for b in Bs:
            for c in Cs:
                loss = 0
                for i, x in enumerate(xs):
                    pred = x ** 2 * a + x * b + c
                    loss += (pred - ys[i]) ** 2
                if loss < minLoss:
                    minLoss = loss
                    best = [a, b, c]
    return best, minLoss
