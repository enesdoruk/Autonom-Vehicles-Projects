import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from LineDetection.calibration_process import calibrate_camera, undistort
from LineDetection.binarization_process import binarize


def birdeye(img, verbose=False):
    h, w = img.shape[:2]
    
    src = np.float32([[595, 465],
                      [50, 465],
                      [235, 370],
                      [415, 370]])
    dst = np.float32([[640, 640],
                      [0, 640],
                      [0, 0],
                      [640, 0]])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)

    return warped, M, Minv



