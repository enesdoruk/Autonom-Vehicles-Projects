import setup_path
import airsim
import cv2
import numpy as np
import os
import time
from Sensors.RGBImage.camera_calibration.calibration_process import calibrate_camera, undistort

def rgb_image_data(client):

    responses_rgb = client.simGetImages([
        airsim.ImageRequest("front_center_rgb", airsim.ImageType.Scene,False,False)
        ])

    ret, mtx, dist, rvecs, tvecs = calibrate_camera(calib_images_dir='/home/enes/AirSim/PythonClient/car/BTUCARF/Sensors/RGBImage/camera_calibration/camera_cal')

    img1d = np.fromstring(responses_rgb[0].image_data_uint8, dtype=np.uint8) 
    img_rgb = img1d.reshape(responses_rgb[0].height, responses_rgb[0].width, 3)

    
    img_undist = undistort(img_rgb, mtx, dist)


    return img_rgb
