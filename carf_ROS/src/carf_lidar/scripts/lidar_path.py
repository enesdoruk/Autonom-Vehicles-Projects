#!/usr/bin/python
# -*- coding: UTF-8 -*-

import rospy
from std_msgs.msg import Float32MultiArray
import serial
import time
from sensor_msgs.msg import Image
import numpy as np
import cv2

def lidarPath(pathImg):
    im = np.frombuffer(pathImg.data, dtype=np.uint8).reshape(pathImg.height, pathImg.width, -1)
    im = cv2.resize(im, (640,640))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3,3), np.uint8)
    img_dilation = cv2.dilate(im, kernel, iterations=1)

    cv2.imshow("pathImg", img_dilation)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        rospy.signal_shutdown('kapatiliyor...')
    
def main():
    rospy.init_node('lidar_path', anonymous=True)

    rospy.Subscriber("/carf_mapImg", Image, lidarPath)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("kapatiliyor")
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
    
