import cv2
import numpy as np


def chart2D(x, y, res=200, UPscale=4, inz=0, fin=1):
    m = np.zeros((res, res, 3))
    for pos, coors in enumerate(x):
        if y[pos, 0] == 0:
            c = (255, 0, 0)
        else:
            c = (0, 0, 255)
        center = (int(res*(coors[0]-inz)/fin), int(res*(coors[1]-inz)/fin))
        cv2.circle(m, center, 6, c, -1)

    cv2.imshow("w1", cv2.resize(m, (res * UPscale, res * UPscale)))
    cv2.waitKey(100000)
    cv2.destroyAllWindows()
