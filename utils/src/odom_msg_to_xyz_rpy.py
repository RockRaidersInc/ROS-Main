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

from nav_msgs.msg import Odometry as Odom_msg
from dependent_messages.msg import XYZRollPitchYaw as RPY_msg

from quaternion_to_rpy import quat_to_rpy


def signal_handler(sig, frame):
    print('control c detected, exiting')
    sys.exit(0)


def callback(in_msg):
    y = in_msg.pose.pose.orientation.x  # swapping x and y is important, otherwise y is the forward direction
    x = in_msg.pose.pose.orientation.y
    z = in_msg.pose.pose.orientation.z
    w = in_msg.pose.pose.orientation.w
    out_msg = RPY_msg()
    # out_msg.header = in_msg.header
    # out_msg.orientation.roll  = atan2(2*y*w + 2*x*z, 1 - 2*y*y - 2*z*z) * 180 / pi
    # out_msg.orientation.pitch = atan2(2*x*w + 2*y*z, 1 - 2*x*x - 2*z*z) * 180 / pi
    # out_msg.orientation.yaw   = asin(2*x*y + 2*z*w) * 180 / pi

    roll, pitch, yaw = quat_to_rpy(in_msg.pose.pose.orientation)
    out_msg.orientation.roll = roll
    out_msg.orientation.pitch = pitch
    out_msg.orientation.yaw = yaw

    out_msg.position.x = in_msg.pose.pose.position.x
    out_msg.position.y = in_msg.pose.pose.position.y
    out_msg.position.z = in_msg.pose.pose.position.z

    pub.publish(out_msg)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)  # so that control-c works
    rospy.init_node("odom_message_quaternion_to_xyz_roll_pitch_yaw")
    sub = rospy.Subscriber("~odom_msg_in", Odom_msg, callback, queue_size = 2)
    pub = rospy.Publisher("~xyz_rpy_msg_out", RPY_msg, queue_size = 2)
    rospy.spin()



