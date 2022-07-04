import numpy as np

def optimizer(data, equation, abc, lr):
    totalLoss = 0
    for x, y in data:
        pred = equation(x, abc)
        loss = (y-pred)**2
        totalLoss += loss

        abc[1] += lr * loss * x
        abc[2] += lr * loss

    print(totalLoss, abc)
    return abc, totalLoss

