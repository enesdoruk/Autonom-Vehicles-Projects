import setup_path
import airsim
import cv2
import numpy as np
import os
import time

def distance_front(client):
    distance_sensor_data_front = client.getDistanceSensorData(distance_sensor_name = "Distance_on", vehicle_name = "")
    print("Distance_sensor_front =====",distance_sensor_data_front.distance)
    return distance_sensor_data_front.distance

def distance_right(client):
    distance_sensor_data_right = client.getDistanceSensorData(distance_sensor_name = "Distance_sag", vehicle_name = "")
    print("Distance_sensor_right =====",distance_sensor_data_right.distance)
    return distance_sensor_data_right.distance

def distance_left(client):
    distance_sensor_data_left = client.getDistanceSensorData(distance_sensor_name = "Distance_sol", vehicle_name = "")
    print("Distance_sensor_left =====",distance_sensor_data_left.distance)
    return distance_sensor_data_left.distance
