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

from geometry_msgs.msg import Vector3Stamped as Vector3Stamped_msg
from sensor_msgs.msg import MagneticField as Mag_msg
from sensor_msgs.msg import Imu as Imu_msg
from dependent_messages.msg import RollPitchYaw as RPY_msg


def signal_handler(sig, frame):
    print('control c detected, exiting')
    sys.exit(0)


def callback(in_msg):
    y = in_msg.orientation.x  # swapping x and y is important, otherwise y is the forward direction
    x = in_msg.orientation.y
    z = in_msg.orientation.z
    w = in_msg.orientation.w
    out_msg = RPY_msg()
    # out_msg.header = in_msg.header
    out_msg.roll  = atan2(2*y*w + 2*x*z, 1 - 2*y*y - 2*z*z) * 180 / pi
    out_msg.pitch = atan2(2*x*w + 2*y*z, 1 - 2*x*x - 2*z*z) * 180 / pi
    out_msg.yaw   = -asin(2*x*y + 2*z*w) * 180 / pi
    pub.publish(out_msg)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)  # so that control-c works
    rospy.init_node("imu_message_quaternion_to_roll_pitch_yaw")
    sub = rospy.Subscriber("~imu_msg_in", Imu_msg, callback, queue_size = 2)
    pub = rospy.Publisher("~rpy_msg_out", RPY_msg, queue_size = 2)
    rospy.spin()



