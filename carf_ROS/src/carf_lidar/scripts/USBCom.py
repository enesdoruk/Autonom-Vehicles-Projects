#!/usr/bin/python
# -*- coding: UTF-8 -*-

import rospy
from std_msgs.msg import Float32MultiArray
import serial
import time
from playsound import playsound
from sensor_msgs.msg import Range
import message_filters

def sendData(dataUsb):
    #distance = dataRadar.range

    angle = dataUsb.data[0]
    angle = angle + 35
    if angle < 0:
        angle = 0

    speed = dataUsb.data[1]    

    #if distance < 1 and distance > 0.1:
    #    speed = 0
    #print("Distance = ", dataRadar.range)

    if angle > 70:
        angle = 70

    print("Angle = ", angle)
    print("speed = ", speed)
    print('*'*40)

    ser = serial.Serial('/dev/ttyACM0', 115200)
    
    arrayAngle = bytearray([int(speed), int(angle)])
    print('array ang =', arrayAngle)
    ser.write(arrayAngle)
    ser.close()
    
def usb_data():
    rospy.init_node('usb_node', anonymous=True)
    usb = rospy.Subscriber("/usb_com", Float32MultiArray, sendData )
    #radar = message_filters.Subscriber("/tfmini_ros_node/TFmini", Range)

    #ts = message_filters.ApproximateTimeSynchronizer([usb, radar],10,0.1,allow_headerless=True)
    #ts.registerCallback(sendData)

    rospy.spin()

if __name__ == '__main__':
    usb_data()
    
