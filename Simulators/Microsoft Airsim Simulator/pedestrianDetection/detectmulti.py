import cv2
import numpy as np
import time
from skimage.feature import hog
from sklearn.externals import joblib
from pedestrianDetection.nms import nms

global scaleFactor
global inverse 
global winStride 
global winSize 

scaleFactor = 1.5
inverse = 1.0/scaleFactor
winStride = (8, 8)
winSize = (128, 64)

def appendRects(i, j, conf, c, rects):
    x = int((j)*pow(scaleFactor, c))
    y = int((i)*pow(scaleFactor, c))
    w = int((64)*pow(scaleFactor, c))
    h = int((128)*pow(scaleFactor, c))
    rects.append((x, y, conf, w, h))

def pedestrian(img,clf):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rects = []

    h, w = gray.shape
    count = 0
    while (h >= 128 and w >= 64):
        h, w= gray.shape
        horiz = w - 64
        vert = h - 128
        print (horiz, vert)
        i = 0
        j = 0
        while i < vert:
            j = 0
            while j < horiz:
                portion = gray[i:i+winSize[0], j:j+winSize[1]]
                features = hog(portion, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2")
                result = clf.predict([features])

                if int(result[0]) == 1:
                    confidence = clf.decision_function([features])
                    appendRects(i, j, confidence, count, rects)

                j = j + winStride[0]

            i = i + winStride[1]

        gray = cv2.resize(gray, (int(w*inverse), int(h*inverse)), interpolation=cv2.INTER_AREA)
        count = count + 1
        print (count)

    nms_rects = nms(rects, 0.2)

    return nms_rects

