#!/usr/bin/env python

import rospy
import serial
from std_msgs.msg import Int16


rospy.init_node('ArduinoEncoderStream')
pub = rospy.Publisher('/encoders/test',Int16, queue_size=10)
publish_rate = rospy.Rate(20) # This should be higher Arduino update rate ot ensure it does not fall behind


ser = serial.Serial('/dev/ttyACM0')
ser.flushInput()
ser.flushOutput()

while not rospy.is_shutdown():
    line = ser.readline()
    if line != '':
        encode = int(line)
        pub.publish(encode)
        publish_rate.sleep()