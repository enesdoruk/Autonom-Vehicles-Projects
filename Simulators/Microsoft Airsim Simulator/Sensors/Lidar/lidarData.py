import setup_path
import airsim
import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib


fig = plt.figure()
ax = plt.axes(projection='3d')

x = []
y = []

def Average(lst): 
    return sum(lst) / len(lst)

def lidar_data(client):
    lidarData = client.getLidarData()
    
    if (len(lidarData.point_cloud) < 3):
        print("\tNo points received from Lidar data")
    
    else:
        points = np.array(lidarData.point_cloud, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0]/3), 3))

        angle = int(len(points)/60)
        liste = []
        for i in range(len(points)):
            liste.append(np.sqrt(points[i,0]**2 + points[i,1]**2 + points[i,2]**2))
        r1 = liste[0:angle*20]
        r2 = liste[angle*20:angle*40]
        r3 = liste[angle*40:]
        
        front_left = 0
        front = 0
        front_right = 0
        
        if len(r1) > 0 and len(r2) > 0 and len(r3) > 0 :
            front_left = min(r1)
            front = min(r2)
            front_right = min(r3)
        
            avg_r1 = Average(r1)
            avg_r2 = Average(r2)
            avg_r3 = Average(r3)
            
            x.append(lidarData.pose.position.x_val)
            y.append(lidarData.pose.position.y_val)

            # print("length points =", len(points))
            # print("1.region: min {}, max {}".format(min(r1), max(r1)))
            # print("2.region: min {}, max {}".format(min(r2), max(r2)))
            # print("3.region: min {}, max {}".format(min(r3), max(r3)))
            # print("front_left: {}, front: {}, front_right: {}".format(front_left, front, front_right))
            # print("length of region_1: ", len(r1))
            # print("length of region_2: ", len(r2))
            # print("length of region_3: ", len(r3))
            # print("lidar position: x {}, y {}, z {}".format(lidarData.pose.position.x_val,lidarData.pose.position.y_val,lidarData.pose.position.z_val))
            print("--"*10)       

        return front_left, front, front_right

def animate(client): 
    ptc = client.getLidarData().point_cloud
    xs = np.empty(shape=[int(len(ptc)/3),])
    ys = np.empty(shape=[int(len(ptc)/3),])
    zs = np.empty(shape=[int(len(ptc)/3),])
    for i in range(0,len(ptc)-1,3):
        xs[int(i/3)]=ptc[i]
        ys[int(i/3)]=ptc[i+1]
        zs[int(i/3)]=-ptc[i+2]
        
    ax.set_zlim([0, 20])
    ax.scatter(xs,ys,zs,zdir='z',s=10,c='b')


