import generateDataset
import visualizer
import numpy as np

nSamples = 400
startingPoint = 0
endPoint = 1

def function(c):
    x, y = c
    if (x-0.5)**2 + (y-0.5)**2 < 0.16:
        return 1
    else:
        return 0



if __name__ == '__main__':
    x, y = generateDataset.datasets().dotChart(nSamples, startingPoint, endPoint, function)
    visualizer.chart2D(x, y, 400, 2, startingPoint, endPoint)




