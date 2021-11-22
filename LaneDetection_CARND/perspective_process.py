import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from calibration_process import calibrate_camera, undistort
from binarization_process import binarize


def birdeye(img, verbose=False):
    h, w = img.shape[:2]
    
    src = np.float32([[w, h-10],
                      [0, h-10],
                      [546, 460],
                      [732, 460]])
    dst = np.float32([[w, h],
                      [0, h],
                      [0, 0],
                      [w, 0]])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)

    return warped, M, Minv



