import setup_path
import airsim
import cv2
import numpy as np
import os
import time

def do_control(client, car_controls, left_distance, right_distance, front_distance,label_sign, dist_sign, label_light, dist_light, label_x, label_y):
    
    serit = 0
    for i in range(len(label_sign)):
        if label_sign[0] == 'serit':
            serit += 1 
    
    if serit == 2:
        start_serit = time.time()
        if time.time() - start < 5: 
            car_controls.throttle = 0.0
            car_controls.steering = 0
            client.setCarControls(car_controls)
            
    if front_distance > 17 and left_distance > 2 and right_distance > 2:
        car_controls.throttle = 0.6
        car_controls.steering = 0
        client.setCarControls(car_controls)
        print("duz")
        
        if label_light[0] == 'stop' and int(dist_light[0]) <= 30:
            car_controls.throttle = 0
            client.setCarControls(car_controls)
            
            for i in range(len(label_sign)):
                if label_sign[i] == 'turnLeft' and dist_sign[i] < 30:
                    car_controls.throttle = 0.6
                    car_controls.steering = -0.2
                    client.setCarControls(car_controls)
                    
                if label_sign[i] == 'turnRight' and dist_sign[i] < 30:
                    car_controls.throttle = 0.4
                    car_controls.steering = 0.4
                    client.setCarControls(car_controls)
                    
                if label_sign[i] == 'station' and dist_sign[i] < 30:
                    car_controls.throttle = 0.0
                    client.setCarControls(car_controls)
                    start = time.time()
                    if time.time() - start > 25:
                        car_controls.throttle = 0.6
                        client.setCarControls(car_controls)
                    
                if label_sign[i] == 'parking':
                    Center_x = 320
                    center_y = 640
                    
                    if (label_x - center_x > 20):
                        car_controls.throttle = 0.4
                        car_controls.steering = -0.2
                        client.setCarControls(car_controls)
                    else:
                        car_controls.throttle = 0.4
                        car_controls.steering = 0.2
                        client.setCarControls(car_controls)
                        
        elif label_light[0] == 'warning' and int(dist_light[0]) <= 30:
            car_controls.throttle = car_controls.throttle // 2
            client.setCarControls(car_controls)
            
            for i in range(len(label_sign)):
                if label_sign[i] == 'turnLeft' and dist_sign[i] < 30:
                    car_controls.throttle = 0.6
                    car_controls.steering = -0.2
                    client.setCarControls(car_controls)
                    
                if label_sign[i] == 'turnRight' and dist_sign[i] < 30:
                    car_controls.throttle = 0.4
                    car_controls.steering = 0.4
                    client.setCarControls(car_controls)
                    
                if label_sign[i] == 'station' and dist_sign[i] < 30:
                    car_controls.throttle = 0.0
                    client.setCarControls(car_controls)
                    start = time.time()
                    if time.time() - start > 25:
                        car_controls.throttle = 0.6
                        client.setCarControls(car_controls)
                    
                if label_sign[i] == 'parking':
                    Center_x = 320
                    center_y = 640
                    
                    if (label_x - center_x > 20):
                        car_controls.throttle = 0.4
                        car_controls.steering = -0.2
                        client.setCarControls(car_controls)
                    else:
                        car_controls.throttle = 0.4
                        car_controls.steering = 0.2
                        client.setCarControls(car_controls) 
                        
        elif label_light[0] == 'go' and int(dist_light[0]) <= 30:
            car_controls.throttle = 0.6
            car_controls.steering = 0
            client.setCarControls(car_controls)
            
            for i in range(len(label_sign)):
                if label_sign[i] == 'turnLeft' and dist_sign[i] < 30:
                    car_controls.throttle = 0.6
                    car_controls.steering = -0.2
                    client.setCarControls(car_controls)
                    
                if label_sign[i] == 'turnRight' and dist_sign[i] < 30:
                    car_controls.throttle = 0.4
                    car_controls.steering = 0.4
                    client.setCarControls(car_controls)
                    
                if label_sign[i] == 'station' and dist_sign[i] < 30:
                    car_controls.throttle = 0.0
                    client.setCarControls(car_controls)
                    start = time.time()
                    if time.time() - start > 25:
                        car_controls.throttle = 0.6
                        client.setCarControls(car_controls)
                    
                if label_sign[i] == 'parking':
                    Center_x = 320
                    center_y = 640
                    
                    if (label_x - center_x > 20):
                        car_controls.throttle = 0.4
                        car_controls.steering = -0.2
                        client.setCarControls(car_controls)
                    else:
                        car_controls.throttle = 0.4
                        car_controls.steering = 0.2
                        client.setCarControls(car_controls)                
    
    elif front_distance > 17 and left_distance < 2 and right_distance > 2:
        car_controls.throttle = 0.6
        car_controls.steering = 0.2
        client.setCarControls(car_controls)
        print("saga meyil")

        if label_light[0] == 'stop' and int(dist_light[0]) <= 30:
            car_controls.throttle = 0
            client.setCarControls(car_controls)
            
            for i in range(len(label_sign)):
                if label_sign[i] == 'turnLeft' and dist_sign[i] < 30:
                    car_controls.throttle = 0.6
                    car_controls.steering = -0.2
                    client.setCarControls(car_controls)
                    
                if label_sign[i] == 'turnRight' and dist_sign[i] < 30:
                    car_controls.throttle = 0.4
                    car_controls.steering = 0.4
                    client.setCarControls(car_controls)
                    
                if label_sign[i] == 'station' and dist_sign[i] < 30:
                    car_controls.throttle = 0.0
                    client.setCarControls(car_controls)
                    start = time.time()
                    if time.time() - start > 25:
                        car_controls.throttle = 0.6
                        client.setCarControls(car_controls)
                    
                if label_sign[i] == 'parking':
                    Center_x = 320
                    center_y = 640
                    
                    if (label_x - center_x > 20):
                        car_controls.throttle = 0.4
                        car_controls.steering = -0.2
                        client.setCarControls(car_controls)
                    else:
                        car_controls.throttle = 0.4
                        car_controls.steering = 0.2
                        client.setCarControls(car_controls)
                        
        elif label_light[0] == 'warning' and int(dist_light[0]) <= 30:
            car_controls.throttle = car_controls.throttle // 2
            client.setCarControls(car_controls)
            
            for i in range(len(label_sign)):
                if label_sign[i] == 'turnLeft' and dist_sign[i] < 30:
                    car_controls.throttle = 0.6
                    car_controls.steering = -0.2
                    client.setCarControls(car_controls)
                    
                if label_sign[i] == 'turnRight' and dist_sign[i] < 30:
                    car_controls.throttle = 0.4
                    car_controls.steering = 0.4
                    client.setCarControls(car_controls)
                    
                if label_sign[i] == 'station' and dist_sign[i] < 30:
                    car_controls.throttle = 0.0
                    client.setCarControls(car_controls)
                    start = time.time()
                    if time.time() - start > 25:
                        car_controls.throttle = 0.6
                        client.setCarControls(car_controls)
                    
                if label_sign[i] == 'parking':
                    Center_x = 320
                    center_y = 640
                    
                    if (label_x - center_x > 20):
                        car_controls.throttle = 0.4
                        car_controls.steering = -0.2
                        client.setCarControls(car_controls)
                    else:
                        car_controls.throttle = 0.4
                        car_controls.steering = 0.2
                        client.setCarControls(car_controls) 
                        
        elif label_light[0] == 'go' and int(dist_light[0]) <= 30:
            car_controls.throttle = 0.6
            car_controls.steering = 0
            client.setCarControls(car_controls)
            
            for i in range(len(label_sign)):
                if label_sign[i] == 'turnLeft' and dist_sign[i] < 30:
                    car_controls.throttle = 0.6
                    car_controls.steering = -0.2
                    client.setCarControls(car_controls)
                    
                if label_sign[i] == 'turnRight' and dist_sign[i] < 30:
                    car_controls.throttle = 0.4
                    car_controls.steering = 0.4
                    client.setCarControls(car_controls)
                    
                if label_sign[i] == 'station' and dist_sign[i] < 30:
                    car_controls.throttle = 0.0
                    client.setCarControls(car_controls)
                    start = time.time()
                    if time.time() - start > 25:
                        car_controls.throttle = 0.6
                        client.setCarControls(car_controls)
                    
                if label_sign[i] == 'parking':
                    Center_x = 320
                    center_y = 640
                    
                    if (label_x - center_x > 20):
                        car_controls.throttle = 0.4
                        car_controls.steering = -0.2
                        client.setCarControls(car_controls)
                    else:
                        car_controls.throttle = 0.4
                        car_controls.steering = 0.2
                        client.setCarControls(car_controls)              
 
    elif front_distance > 17 and left_distance > 2 and right_distance < 2:
        car_controls.throttle = 0.6
        car_controls.steering = -0.2
        client.setCarControls(car_controls)
        print("sola meyil")
    
        if label_light[0] == 'stop' and int(dist_light[0]) <= 30:
            car_controls.throttle = 0
            client.setCarControls(car_controls)
            
            for i in range(len(label_sign)):
                if label_sign[i] == 'turnLeft' and dist_sign[i] < 30:
                    car_controls.throttle = 0.6
                    car_controls.steering = -0.2
                    client.setCarControls(car_controls)
                    
                if label_sign[i] == 'turnRight' and dist_sign[i] < 30:
                    car_controls.throttle = 0.4
                    car_controls.steering = 0.4
                    client.setCarControls(car_controls)
                    
                if label_sign[i] == 'station' and dist_sign[i] < 30:
                    car_controls.throttle = 0.0
                    client.setCarControls(car_controls)
                    start = time.time()
                    if time.time() - start > 25:
                        car_controls.throttle = 0.6
                        client.setCarControls(car_controls)
                    
                if label_sign[i] == 'parking':
                    Center_x = 320
                    center_y = 640
                    
                    if (label_x - center_x > 20):
                        car_controls.throttle = 0.4
                        car_controls.steering = -0.2
                        client.setCarControls(car_controls)
                    else:
                        car_controls.throttle = 0.4
                        car_controls.steering = 0.2
                        client.setCarControls(car_controls)
                        
        elif label_light[0] == 'warning' and int(dist_light[0]) <= 30:
            car_controls.throttle = car_controls.throttle // 2
            client.setCarControls(car_controls)
            
            for i in range(len(label_sign)):
                if label_sign[i] == 'turnLeft' and dist_sign[i] < 30:
                    car_controls.throttle = 0.6
                    car_controls.steering = -0.2
                    client.setCarControls(car_controls)
                    
                if label_sign[i] == 'turnRight' and dist_sign[i] < 30:
                    car_controls.throttle = 0.4
                    car_controls.steering = 0.4
                    client.setCarControls(car_controls)
                    
                if label_sign[i] == 'station' and dist_sign[i] < 30:
                    car_controls.throttle = 0.0
                    client.setCarControls(car_controls)
                    start = time.time()
                    if time.time() - start > 25:
                        car_controls.throttle = 0.6
                        client.setCarControls(car_controls)
                    
                if label_sign[i] == 'parking':
                    Center_x = 320
                    center_y = 640
                    
                    if (label_x - center_x > 20):
                        car_controls.throttle = 0.4
                        car_controls.steering = -0.2
                        client.setCarControls(car_controls)
                    else:
                        car_controls.throttle = 0.4
                        car_controls.steering = 0.2
                        client.setCarControls(car_controls) 
                        
        elif label_light[0] == 'go' and int(dist_light[0]) <= 30:
            car_controls.throttle = 0.6
            car_controls.steering = 0
            client.setCarControls(car_controls)
            
            for i in range(len(label_sign)):
                if label_sign[i] == 'turnLeft' and dist_sign[i] < 30:
                    car_controls.throttle = 0.6
                    car_controls.steering = -0.2
                    client.setCarControls(car_controls)
                    
                if label_sign[i] == 'turnRight' and dist_sign[i] < 30:
                    car_controls.throttle = 0.4
                    car_controls.steering = 0.4
                    client.setCarControls(car_controls)
                    
                if label_sign[i] == 'station' and dist_sign[i] < 30:
                    car_controls.throttle = 0.0
                    client.setCarControls(car_controls)
                    start = time.time()
                    if time.time() - start > 25:
                        car_controls.throttle = 0.6
                        client.setCarControls(car_controls)
                    
                if label_sign[i] == 'parking':
                    Center_x = 320
                    center_y = 640
                    
                    if (label_x - center_x > 20):
                        car_controls.throttle = 0.4
                        car_controls.steering = -0.2
                        client.setCarControls(car_controls)
                    else:
                        car_controls.throttle = 0.4
                        car_controls.steering = 0.2
                        client.setCarControls(car_controls)              
    
    elif front_distance < 10:
        car_controls.throttle = 0.0
        car_controls.steering = -0.0
        client.setCarControls(car_controls)