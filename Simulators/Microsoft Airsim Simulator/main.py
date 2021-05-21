import setup_path
import airsim
import cv2
import numpy as np
import os
import time
import torch
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import matplotlib
from sklearn.externals import joblib

import warnings
warnings.filterwarnings("ignore")

from objectDetection.models.experimental import attempt_load
from objectDetection.utils.datasets import LoadStreams, LoadImages, letterbox
from objectDetection.utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from objectDetection.utils.plots import plot_one_box
from objectDetection.utils.torch_utils import select_device, load_classifier, time_synchronized
from objectDetection.detect import detect_object

from LightDetect.detect_light import light_detect

from Sensors.Distance import distanceSensor
from Sensors.Lidar import lidarData
from Sensors.RGBImage import rgb_image
from Sensors.DepthImage import depth_image

from Control.changeRoad import do_control
from Control.parking import parking_control

from LineDetection import line_detect
from LineDetection import perspective_process

from pedestrianDetection.detectmulti import pedestrian

matplotlib.use('TkAgg')

client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
print("API Control enabled: %s" % client.isApiControlEnabled())

car_controls = airsim.CarControls()

camera_pose = airsim.Pose(airsim.Vector3r(0.7, 0, -3), airsim.to_quaternion(0.2, 0, 0))  
client.simSetCameraPose(0, camera_pose)

client.simEnableWeather(True)
#client.simSetTimeOfDay(True, start_datetime = "", is_start_datetime_dst = False, celestial_clock_speed = 1, update_interval_secs = 90, move_sun = True)
#client.simSetWeatherParameter(airsim.WeatherParameter.Fog, 0.1)


#pedestrian_pkl = joblib.load("./pedestrianDetection/person_final.pkl")

def objectDetect(RGB, DEPTH):

    img,pred,names,colors = detect_object(RGB)
    labelobj_list = []
    distobj_list = []
    label_x = []
    label_y = []

    for i, det in enumerate(pred):  
        s, im0 =  '', RGB
        s += '%gx%g ' % img.shape[2:]  
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]] 

        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            for *xyxy, conf, cls in reversed(det):

                label = f'{names[int(cls)]}'

                x1 = int(((xyxy[2].item() - xyxy[0].item())/2 + xyxy[0])/2) 
                y1 = int(((xyxy[3].item() - xyxy[1].item())/2 + xyxy[1])/2)
                disobj = depth_image.measure_distance_object(DEPTH,y1,x1)
                #print("{} distance from camera {} m".format(label,disobj))

                cv2.line(im0,(int((xyxy[2].item() - xyxy[0].item())/2)+xyxy[0],xyxy[1]),(int((xyxy[2].item() - xyxy[0].item())/2)+xyxy[0],xyxy[3]),(colors[int(cls)]),1)
                cv2.line(im0,(xyxy[0],int(((xyxy[3].item() - xyxy[1].item())/2 + xyxy[1]))),(xyxy[2],int(((xyxy[3].item() - xyxy[1].item())/2 + xyxy[1]))),(colors[int(cls)]),1)
                cv2.circle(im0, (x1*2,y1*2), 1, (colors[int(cls)]), 2)
                cv2.putText(im0,"distance: {} m".format(disobj), (xyxy[0],xyxy[3]+8), cv2.FONT_HERSHEY_SIMPLEX, 0.3, colors[int(cls)], 1, cv2.LINE_AA)
                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                
                if disobj < 60:
                #    print("Label {}, Distance {}".format(label, disobj))
                    labelobj_list.append(label)
                    distobj_list.append(disobj)
                    label_x.append(x1)
                    label_y.append(y1)
    
    return labelobj_list, distobj_list, label_x, label_y

        
def traffic_light(RGB, DEPTH):

    img,pred,names,colors = light_detect(RGB)
    label_list = []
    dist_list = []


    for i, det in enumerate(pred):  
        s, im0 =  '', RGB
        s += '%gx%g ' % img.shape[2:]  
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]] 

        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            for *xyxy, conf, cls in reversed(det):

                label = f'{names[int(cls)]}'

                x1 = int(((xyxy[2].item() - xyxy[0].item())/2 + xyxy[0])/2) 
                y1 = int(((xyxy[3].item() - xyxy[1].item())/2 + xyxy[1])/2)
                disobj = depth_image.measure_distance_object(DEPTH,y1,x1)
                #print("{} distance from camera {} m".format(label,disobj))

                cv2.line(im0,(int((xyxy[2].item() - xyxy[0].item())/2)+xyxy[0],xyxy[1]),(int((xyxy[2].item() - xyxy[0].item())/2)+xyxy[0],xyxy[3]),(colors[int(cls)]),1)
                cv2.line(im0,(xyxy[0],int(((xyxy[3].item() - xyxy[1].item())/2 + xyxy[1]))),(xyxy[2],int(((xyxy[3].item() - xyxy[1].item())/2 + xyxy[1]))),(colors[int(cls)]),1)
                cv2.circle(im0, (x1*2,y1*2), 1, (colors[int(cls)]), 2)
                cv2.putText(im0,"distance: {} m".format(disobj), (xyxy[0],xyxy[3]+8), cv2.FONT_HERSHEY_SIMPLEX, 0.3, colors[int(cls)], 1, cv2.LINE_AA)
                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                
                if disobj < 40:
                #    print("Label {}, Distance {}".format(label, disobj))
                    label_list.append(label)
                    dist_list.append(disobj)

                
    return label_list, dist_list

serit_time = 0
station_time = 0
sayac = 0 

prev_frame_time = 0
new_frame_time = 0

while (cv2.waitKey(1) & 0xFF) == 0xFF:
    print('#'*80)
    
    new_frame_time = time.time()
    
    car_state = client.getCarState()

    left_distance = distanceSensor.distance_left(client)
    right_distance = distanceSensor.distance_right(client)
    front_distance = distanceSensor.distance_front(client)
   
    lidarData.lidar_data(client)
    
    DEPTH = depth_image.depth_image_data(client)
    RGB = rgb_image.rgb_image_data(client)

    #pedestrian_coord = pedestrian(RGB2, pedestrian_pkl)
    #for (a, b, conf, c, d) in pedestrian_coord:
    #    cv2.rectangle(RGB, (a, b), (a+c, b+d), (0, 255, 0), 2)
     
    label_sign, dist_sign, label_x, label_y = objectDetect(RGB, DEPTH)
    print("Label sign {}, Dist sign {}".format(label_sign, dist_sign)) 
    label_light, dist_light = traffic_light(RGB, DEPTH)
    print("Label light {}, Dist light {}".format(label_light, dist_light))
    
     
    if 'serit' in label_sign:
        if serit_time < 1:
            for i in range(len(label_sign)):
                if label_sign[i] == 'serit' and dist_sign[i] < 30:
                    start = time.time()
                    while time.time() - start < 5:
                        car_controls.throttle = 0.0
                        car_controls.steering = 0.0
                        client.setCarControls(car_controls)
                    serit_time += 1
    if serit_time == 1:
        parking_control(client, car_controls, left_distance, right_distance, front_distance,label_sign, dist_sign, label_light, dist_light, label_x, label_y)
    else:
        do_control(client, car_controls,label_sign, dist_sign, label_light, dist_light, label_x, label_y)
     
    if 'station' in label_sign:
        print("station algilandi")
        if station_time < 1:
            for i in range(len(label_sign)):
                if label_sign[i] == 'station':
                    print("station indisi algilandi")
                    if int(dist_sign[i]) < 30:
                        print("durma islemine baslayacak")
                        car_controls.throttle = 0.0
                        car_controls.steering = 0.0
                        client.setCarControls(car_controls)
                        
                        start = time.time()
                        while time.time() - start < 10:
                            car_controls.throttle = 0.0
                            car_controls.steering = 0.0
                            client.setCarControls(car_controls) 
                            cv2.putText(RGB, ''.format(int(time.time() - start)), (320, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 1, cv2.LINE_AA)
                        station_time += 1
        
    sayac += 1
    if sayac == 20:
        serit_time = 0
        station_time = 0
    
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    
    # try:
        # line_dtct, Wpts_x, Wpts_y = line_detect.process_pipeline(RGB, keep_state=False)
        # cv2.putText(line_dtct, 'FPS: {:.4f}'.format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 1, cv2.LINE_AA)
        #print("x= ",Wpts_x, "y= ",Wpts_y)
        # cv2.imshow("CARF", line_dtct)
        # cv2.imshow("depth", DEPTH)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
            # break
                
    #except:
    cv2.putText(RGB, 'FPS: {:.4f}'.format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow("CARF",RGB)
    #cv2.imshow("depth", DEPTH)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
client.reset()
client.enableApiControl(False)