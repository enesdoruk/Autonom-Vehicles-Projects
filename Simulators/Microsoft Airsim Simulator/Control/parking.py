import setup_path
import airsim
import cv2
import numpy as np
import os
import time
import math

def parking_control(client, car_controls, left_distance, right_distance, front_distance,label_sign, dist_sign, label_light, dist_light, label_x, label_y):
    
    if 'serit' in label_sign:
        client.setCarControls(car_controls)
        car_controls.throttle = 0.4
        car_controls.steering = 0
        client.setCarControls(car_controls)
        print("serit kismini gecene kadar ilerle")
        
        if front_distance > 10 and left_distance > 3 and right_distance < 2.2:
            print("hafif sola meyil")
            car_controls.throttle = 0.3
            car_controls.steering = -0.2
            client.setCarControls(car_controls)
        if front_distance > 10 and left_distance > 3 and right_distance > 2.2:    
            print("Bos alan park alani gorene kadar ilerle")
            car_controls.throttle = 0.3
            car_controls.steering = 0
            client.setCarControls(car_controls) 
        
    if front_distance < 12:
        car_controls.throttle = 0.0
        car_controls.steering = 0.0
        client.setCarControls(car_controls)
        print("bir")
    else:
        if label_sign is not None:
            for i in range(len(label_sign)):
                if label_sign[i] != 'parking':  
                    if front_distance > 10 and left_distance > 3 and right_distance < 2.2:
                        print("hafif sola meyil")
                        car_controls.throttle = 0.3
                        car_controls.steering = -0.2
                        client.setCarControls(car_controls)
                    if front_distance > 10 and left_distance > 3 and right_distance > 2.2:    
                        print("Bos alan park alani gorene kadar ilerle")
                        car_controls.throttle = 0.3
                        car_controls.steering = 0
                        client.setCarControls(car_controls)         
                if label_sign[i] == 'parking':   
                    if front_distance > 10 and left_distance > 3 and right_distance < 2.2:
                        print("hafif sola meyil")
                        car_controls.throttle = 0.3
                        car_controls.steering = -0.2
                        client.setCarControls(car_controls)
                    if front_distance > 10 and left_distance > 3 and right_distance > 2.2:    
                        print("Bos alan park alani gorene kadar ilerle")
                        car_controls.throttle = 0.3
                        car_controls.steering = 0
                        client.setCarControls(car_controls)         
                    
                    print("park alani algilandi")
                    object_x = int(label_x[i])
                    object_y = int(label_y[i])
                    dist = dist_sign[i]
                    car_x = 320
                    car_y = 600
                    print("x ekseni ============", object_x)
                    print("y ekseni ============", object_y)
                        
                    oklid  = int(np.sqrt((object_x-car_x)**2 + (car_y-object_y)**2))
                    print("Oklid uzunlugu:", int(oklid))
                        
                    angle_x = np.abs(object_x - car_x)
                    angle_y = np.abs(object_y - car_y)
                    angle_r = math.atan2(angle_y, angle_x)
                    angle_d = math.degrees(angle_r)
                    print("Angle: ", angle_d)
                        
                    if angle_d < 70: 
                        car_controls.throttle = 0.25
                        car_controls.steering = -0.12
                        print("Park alani algilandi sola meyil ediyor")
                        client.setCarControls(car_controls)
                        
                    if angle_d > 70: 
                        car_controls.throttle = 0.25
                        car_controls.steering = 0.12
                        print("Park alani algilandi saga meyil ediyor")
                        client.setCarControls(car_controls)
                        
     