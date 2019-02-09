#!/usr/bin/env python
import rospy
import signal
import sys

from geometry_msgs.msg import Vector3Stamped as Vector3Stamped_msg
from sensor_msgs.msg import MagneticField as Mag_msg
from sensor_msgs.msg import Imu as Imu_msg


def signal_handler(sig, frame):
    print('control c detected, exiting')
    sys.exit(0)


def set_covariance_as_identity(val):
    for i in range(9):
        val[i] = 0
    val[0] = 1
    val[4] = 1
    val[8] = 1


def mag_callback(in_msg):
    out_msg = Mag_msg()
    out_msg.header = in_msg.header
    out_msg.magnetic_field = in_msg.vector
    set_covariance_as_identity(out_msg.magnetic_field_covariance)
    mag_publisher.publish(out_msg)


def imu_callback(in_msg):
    in_msg.header.frame_id = "imu0_link"
    imu_publisher.publish(in_msg)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)  # so that control-c works
    rospy.init_node('magnetometer_republisher')

    mag_subscriber = rospy.Subscriber('~vect3_msg', Vector3Stamped_msg, mag_callback, queue_size = 2)
    mag_publisher = rospy.Publisher('~mag_msg', Mag_msg, queue_size = 2)

    imu_subscriber = rospy.Subscriber('~imu_msg_in', Imu_msg, imu_callback, queue_size = 2)
    imu_publisher = rospy.Publisher('~imu_msg_out', Imu_msg, queue_size = 2)

    rospy.spin()
