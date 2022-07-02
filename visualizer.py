import numpy as np
import cv2

def color(choice):
    print(choice)
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

def chart2D(coords, colors, res=200, UPscale=4, inz=0, fin=1):
    m = np.zeros((res, res, 3))
    for pos, coord in enumerate(coords):
        c = color(colors[pos])
        center = (int(res*(coord[0]-inz)/fin), int(res*(coord[1]-inz)/fin))
        cv2.circle(m, center, 6, c, -1)

    cv2.imshow("w1", cv2.resize(m, (res * UPscale, res * UPscale)))
    cv2.waitKey(100000)
    cv2.destroyAllWindows()
