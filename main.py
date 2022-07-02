import generateDataset
import visualizer
import numpy as np

nSamples = 40
startingPoint = 0
endPoint = 1

def function(c):
    x, y = c
    if (x-0.5)**2 + (y-0.5)**2 < 0.16: return 1
    else: return 0

def equation(n, abc):
    a, b, c = abc
    return 1 - (a*(n**2) + b*n + c)


if __name__ == '__main__':
    dg = generateDataset.datasets()
    #x, y = dg.binClassifier(nSamples, function, startingPoint, endPoint)
    x, y = dg.parabola(nSamples, d=10)
    print(x)
    chart = visualizer.chart2D(x, y, dotSize=2)
    chart = visualizer.chartFunc(chart, equation, (1, 0, 0), dotSize=0)
    visualizer.display(chart)
    model = [1, 0]  # m = 1, q = 0





