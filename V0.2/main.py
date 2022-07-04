import numpy as np
import cv2
import random
import algos

np.set_printoptions(suppress=True)

def functRand(x):
    global a, b, c
    return a * x ** 2 + b * x + c


def funct(x):
    a, b, c = 0.5, 0, 0.3
    return a * x ** 2 + b * x + c

for i in range(1000):
    a, b, c = random.random() * 4, random.random() * 4-2, random.random() - 0.3
    lr = algos.linearRegression(200, 0, 1)
    lr.makedataset(functRand, 0.05)
    lr.chart(200, 1)
    lr.show(800, 10)
    t = lr.train(360, 100)  # n*m*q complexity
    lr.drawPredictions()
    if lr.show(800, 0) == 27: break
    print(f"Computed {lr.n*lr.msamples*lr.qsamples} equations in {t} with parameters {a, b, c}")
cv2.destroyAllWindows()

