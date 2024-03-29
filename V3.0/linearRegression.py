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

mres, qres = 15, 15

for i in range(1000):
    # Make dataset
    lr = algos.linearRegression(200, 0, 1)
    while True:
        a, b, c = random.random() * 4, random.random() * 4 - 2, random.random() - 0.3
        lr.makedataset(functRand, 0.05)
        if lr.x.shape[0] > lr.n/5: break
    # Display dots
    lr.chart(200, 1)
    lr.show(800, 10)

    # Train
    t, pred, loss = lr.train(mres, qres, 8, 0, draw=1)  # n*m*q complexity draw = 0, 1, 2
    print(f"R2 = {round(lr.Rsquared(), 3)}, Loss: {loss[-1]}, m: {round(pred[0], 2)},"
          f" q: {round(pred[1], 2)} with parameters {a, b, c}")

    print(lr.predict(np.array([0.1, 0.2, 0.3, 0.4, 0.5])))
    # R2 is how much better is my function at explaining the distribution than the mean,
    # even low values (<.3) mean that the code is working fine

    #lr.drawPredictions()
    if lr.show(800, 0) == 27: break


cv2.destroyAllWindows()

