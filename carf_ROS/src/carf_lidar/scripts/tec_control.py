#!/usr/bin/python
# -*- coding: UTF-8 -*-

import rospy
from std_msgs.msg import Float32MultiArray
import serial
import time
from sensor_msgs.msg import Range
import message_filters

def tec_control(frontRadar, rightRadar):
    distance_front = frontRadar.range
    distance_right = rightRadar.range

    speed = 1   

    if distance_front < 0.6 and distance_front > 0.1:
        speed = 0
    elif distance_right < 0.6 and distance_right > 0.1:
        speed = 0

    print("front_distance = ", distance_front)
    print("right_distance = ", distance_right)

    ser = serial.Serial('/dev/ttyACM0', 115200)
    
    arrayAngle = bytearray([int(speed), int(35)])

    ser.write(arrayAngle)
    ser.close()
    
def usb_data():
    rospy.init_node('tec_control', anonymous=True)

    frontRadar = message_filters.Subscriber("/tfmini_ros_front/TFmini", Range)
    rightRadar = message_filters.Subscriber("/tfmini_ros_right/TFmini", Range)

    ts = message_filters.ApproximateTimeSynchronizer([frontRadar, rightRadar],10,0.1,allow_headerless=True)
    ts.registerCallback(tec_control)

    rospy.spin()

if __name__ == '__main__':
    usb_data()
    
