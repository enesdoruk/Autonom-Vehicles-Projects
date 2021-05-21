import setup_path
import airsim
import cv2
import numpy as np
import os
import time
import math
from Sensors.Distance import distanceSensor


def do_control(client, car_controls,label_sign, dist_sign, label_light, dist_light, label_x, label_y):
    
    
    l_distance = distanceSensor.distance_left(client)
    r_distance = distanceSensor.distance_right(client)
    front_distance = distanceSensor.distance_front(client)
    print("on====",front_distance)
    print("sol====",l_distance)
    print("sag====",r_distance)
    
    if front_distance < 2:
        print("Dur")
        car_controls.throttle = 0.0
        car_controls.steering = 0.0
        client.setCarControls(car_controls)
    
    else:
        
        if 'stop' in label_light:
            for i in range(len(label_light)):
                if label_light[i] == 'stop':
                    if dist_light[i] < 35:
                        print("Duz gidiyor")
                        car_controls.throttle = 0.0
                        car_controls.steering = 0
                        client.setCarControls(car_controls)
        else:
            if front_distance > 15 and l_distance > 3 and r_distance > 3:
                print("Duz gidiyor")
                car_controls.throttle = 0.32
                car_controls.steering = 0
                client.setCarControls(car_controls)
                
                if 'turnRight' in label_sign:
                    for i in range(len(label_sign)):
                        if label_sign[i] == 'turnRight':
                            if dist_sign[i] < 25:
                                start_time = time.time()
                                while time.time() - start_time < 7:
                                    car_controls.throttle = 0.25
                                    car_controls.steering = 0.11
                                    client.setCarControls(car_controls)
                                start_time2 = time.time()
                                while time.time() - start_time2 < 4:
                                    car_controls.throttle = 0.25
                                    car_controls.steering = 0.3
                                    client.setCarControls(car_controls)
                                    
                                 
                                        
                if 'turnLeft' in label_sign:
                    for i in range(len(label_sign)):
                        if label_sign[i] == 'turnLeft':
                            if dist_sign[i] < 25:
                                start_time = time.time()
                                while time.time() - start_time < 5:
                                    car_controls.throttle = 0.3
                                    car_controls.steering = -0.11
                                    client.setCarControls(car_controls)
                                start_time2 = time.time()
                                while time.time() - start_time2 < 3:
                                    car_controls.throttle = 0.25
                                    car_controls.steering = -0.35
                                    client.setCarControls(car_controls)
                                  
             
            if front_distance > 15 and l_distance < 3 and r_distance > 3:
                print("hafif saga meyil")
                car_controls.throttle = 0.3
                car_controls.steering = 0.1
                client.setCarControls(car_controls)
                
                if 'turnRight' in label_sign:
                    for i in range(len(label_sign)):
                        if label_sign[i] == 'turnRight':
                            if dist_sign[i] < 25:
                                start_time = time.time()
                                while time.time() - start_time < 7:
                                    car_controls.throttle = 0.25
                                    car_controls.steering = 0.10
                                    client.setCarControls(car_controls)
                                start_time2 = time.time()
                                while time.time() - start_time2 < 4:
                                    car_controls.throttle = 0.25
                                    car_controls.steering = 0.3
                                    client.setCarControls(car_controls)
                                 
              
                if 'turnLeft' in label_sign:
                    for i in range(len(label_sign)):
                        if label_sign[i] == 'turnLeft':
                            if dist_sign[i] < 25:
                                start_time = time.time()
                                while time.time() - start_time < 5:
                                    car_controls.throttle = 0.3
                                    car_controls.steering = -0.11
                                    client.setCarControls(car_controls)
                                start_time2 = time.time()
                                while time.time() - start_time2 < 3:
                                    car_controls.throttle = 0.25
                                    car_controls.steering = -0.35
                                    client.setCarControls(car_controls)
                                    
                               
            
            if front_distance > 15 and l_distance > 3 and r_distance < 3:
                print("hafif sola meyil")
                car_controls.throttle = 0.3
                car_controls.steering = -0.1
                client.setCarControls(car_controls)
                
                if 'turnRight' in label_sign:
                    for i in range(len(label_sign)):
                        if label_sign[i] == 'turnRight':
                            if dist_sign[i] < 25:
                                start_time = time.time()
                                while time.time() - start_time < 7:
                                    car_controls.throttle = 0.25
                                    car_controls.steering = 0.10
                                    client.setCarControls(car_controls)
                                start_time2 = time.time()
                                while time.time() - start_time2 < 4:
                                    car_controls.throttle = 0.25
                                    car_controls.steering = 0.3
                                    client.setCarControls(car_controls)
                                 
                                        
                if 'turnLeft' in label_sign:
                    for i in range(len(label_sign)):
                        if label_sign[i] == 'turnLeft':
                            if dist_sign[i] < 25:
                                start_time = time.time()
                                while time.time() - start_time < 5:
                                    car_controls.throttle = 0.3
                                    car_controls.steering = -0.11
                                    client.setCarControls(car_controls)
                                start_time2 = time.time()
                                while time.time() - start_time2 < 3:
                                    car_controls.throttle = 0.25
                                    car_controls.steering = -0.35
                                    client.setCarControls(car_controls)
                                         
                
            if front_distance < 20 and (l_distance > r_distance * 2.2 ):
                print("keskin sola meyil")
                car_controls.throttle = 0.3
                car_controls.steering = -0.47
                client.setCarControls(car_controls)
                
                if 'turnRight' in label_sign:
                    for i in range(len(label_sign)):
                        if label_sign[i] == 'turnRight':
                            if dist_sign[i] < 25:
                                start_time = time.time()
                                while time.time() - start_time < 7:
                                    car_controls.throttle = 0.25
                                    car_controls.steering = 0.11
                                    client.setCarControls(car_controls)
                                start_time2 = time.time()
                                while time.time() - start_time2 < 4:
                                    car_controls.throttle = 0.25
                                    car_controls.steering = 0.3
                                    client.setCarControls(car_controls)
                                                          
                
                if 'turnLeft' in label_sign:
                    for i in range(len(label_sign)):
                        if label_sign[i] == 'turnLeft':
                            if dist_sign[i] < 25:
                                start_time = time.time()
                                while time.time() - start_time < 5:
                                    car_controls.throttle = 0.3
                                    car_controls.steering = -0.11
                                    client.setCarControls(car_controls)
                                start_time2 = time.time()
                                while time.time() - start_time2 < 3:
                                    car_controls.throttle = 0.25
                                    car_controls.steering = -0.35
                                    client.setCarControls(car_controls)
                                                            
            
            if front_distance < 15 and (l_distance * 2.2 < r_distance):
                print("keskin saga meyil")
                car_controls.throttle = 0.3
                car_controls.steering = 0.45
                client.setCarControls(car_controls)
                
                if 'turnRight' in label_sign:
                    for i in range(len(label_sign)):
                        if label_sign[i] == 'turnRight':
                            if dist_sign[i] < 25:
                                start_time = time.time()
                                while time.time() - start_time < 7:
                                    car_controls.throttle = 0.25
                                    car_controls.steering = 0.11
                                    client.setCarControls(car_controls)
                                start_time2 = time.time()
                                while time.time() - start_time2 < 4:
                                    car_controls.throttle = 0.25
                                    car_controls.steering = 0.3
                                    client.setCarControls(car_controls)
                               
            
                if 'turnLeft' in label_sign:
                    for i in range(len(label_sign)):
                        if label_sign[i] == 'turnLeft':
                            if dist_sign[i] < 25:
                                start_time = time.time()
                                while time.time() - start_time < 5:
                                    car_controls.throttle = 0.3
                                    car_controls.steering = -0.11
                                    client.setCarControls(car_controls)
                                start_time2 = time.time()
                                while time.time() - start_time2 < 3:
                                    car_controls.throttle = 0.25
                                    car_controls.steering = -0.35
                                    client.setCarControls(car_controls)
                                