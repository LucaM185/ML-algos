import generateDataset
import visualizer
import cv2
import algos

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

EPOCHS = 300
a, b, c = 0, 0, 0

if __name__ == '__main__':
    dg = generateDataset.datasets()
    #data, labels = dg.binClassifier(nSamples, function, startingPoint, endPoint)
    data, labels = dg.parabola(nSamples, d=10)

    chart = visualizer.chart2D(data, labels, dotSize=2)
    chart = visualizer.chartFunc(chart, equation, [a, b, c], dotSize=0)
    abc, loss = algos.optimizer(data, equation, [a, b, c], 0.01)
    a, b, c = abc
    k = visualizer.display(chart)

    cv2.destroyAllWindows()

