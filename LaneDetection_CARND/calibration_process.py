import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import os.path as path
import pickle

def lazy_calibration(func):
    calibration_cache = 'camera_cal/calibration_data.pickle'

    def wrapper(*args, **kwargs):
        if path.exists(calibration_cache):
            with open(calibration_cache, 'rb') as dump_file:
                calibration = pickle.load(dump_file)
        else:
            calibration = func(*args, **kwargs)
            with open(calibration_cache, 'wb') as dump_file:
                pickle.dump(calibration, dump_file)
        return calibration

    return wrapper


@lazy_calibration
def calibrate_camera(calib_images_dir, verbose=False):
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []

    images = glob.glob(path.join(calib_images_dir, 'calibration*.jpg'))

    for filename in images:
        img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        pattern_found, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        if pattern_found is True:
            objpoints.append(objp)
            imgpoints.append(corners)

    if verbose:
        cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return ret, mtx, dist, rvecs, tvecs

def undistort(frame, mtx, dist, verbose=False):

    frame_undistorted = cv2.undistort(frame, mtx, dist, newCameraMatrix=mtx)
    return frame_undistorted

