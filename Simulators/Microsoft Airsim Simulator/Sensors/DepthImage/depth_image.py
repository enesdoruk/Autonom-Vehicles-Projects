import setup_path
import airsim
import cv2
import numpy as np
import os
import time

def depth_image_data(client):
    
    responses_depth = client.simGetImages([
        airsim.ImageRequest("front_center_depth", airsim.ImageType.DepthPlanner, pixels_as_float=True)
        ]) 

    response = responses_depth[0]

    img1d = np.array(response.image_data_float, dtype=np.float)
    img1d = img1d*3.5+30 
    img1d[img1d>255] = 255
    img2d = np.reshape(img1d, (responses_depth[0].height, responses_depth[0].width))

    depth = np.array(img2d,dtype=np.uint8)

    return depth 

def measure_distance_object(depth_image, x,y):
    
    piksel = depth_image[x,y]
    distance =  int((piksel * 100)/255)

    return distance
