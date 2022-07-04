import numpy as np
import cv2

def color(choice):
    x = np.argmax(choice)
    if x == 0:
        return (255, 0, 0)
    elif x == 1:
        return (0, 0, 255)
    elif x == 2:
        return (0, 255, 0)
    elif x == 3:
        return (255, 255, 0)
    elif x == 4:
        return (255, 0, 255)
    elif x == 5:
        return (0, 255, 255)
    elif x == 6:
        return (255, 255, 255)

def chart2D(coords, colors, res=200, inz=0, fin=1, dotSize=4):
    img = np.zeros((res, res, 3))
    for pos, coord in enumerate(coords):
        c = color(colors[pos])
        center = (int(res*(coord[0]-inz)/fin), int(res*(coord[1]-inz)/fin))
        cv2.circle(img, center, dotSize, c, -1)
    return img



def chartFunc(img, func, params, dotSize=4, res=200):
    for x in np.arange(0, 1, 1/400):
        y = func(x, params)
        cv2.circle(img, (int(x*res), int(y*res)), dotSize, (127, 127, 127), -1)
    return img


def display(img, res=200, UPscale=4):
    cv2.imshow("w1", cv2.resize(img, (res * UPscale, res * UPscale)))
    return cv2.waitKey(10)