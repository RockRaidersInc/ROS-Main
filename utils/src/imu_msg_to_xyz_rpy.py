#!/usr/bin/env python

"""
This node turns the quaternion in IMU messages into roll, pitch, and yaw in degrees.
"""

import rospy
import signal
import sys
from math import atan2
from math import asin
from math import pi

from sensor_msgs.msg import Imu as Imu_msg
from dependent_messages.msg import imu_degrees as Imu_degrees_msg

from quaternion_to_rpy import quat_to_rpy


def signal_handler(sig, frame):
    print('control c detected, exiting')
    sys.exit(0)


def callback(in_msg):
    out_msg = Imu_degrees_msg()

    roll, pitch, yaw = quat_to_rpy(in_msg.orientation)
    out_msg.orientation.roll = roll
    out_msg.orientation.pitch = pitch
    out_msg.orientation.yaw = yaw

    out_msg.angular_velocity = in_msg.angular_velocity
    out_msg.linear_acceleration = in_msg.linear_acceleration

    pub.publish(out_msg)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)  # so that control-c works
    rospy.init_node("odom_message_quaternion_to_xyz_roll_pitch_yaw")
    sub = rospy.Subscriber("~imu_msg_in", Imu_msg, callback, queue_size = 2)
    pub = rospy.Publisher("~imu_msg_out", Imu_degrees_msg, queue_size = 2)
    rospy.spin()

