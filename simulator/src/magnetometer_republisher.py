#!/usr/bin/env python
import rospy
import signal
import sys

from geometry_msgs.msg import Vector3Stamped as Vector3Stamped_msg
from sensor_msgs.msg import MagneticField as Mag_msg
from sensor_msgs.msg import Imu as Imu_msg

from geometry_msgs.msg import Quaternion
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import math


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
    in_msg.header.frame_id = "base_imu_link"
    x, y = in_msg.linear_acceleration.x, in_msg.linear_acceleration.y
    in_msg.linear_acceleration.x = y
    in_msg.linear_acceleration.y = -x

    # gazebo uses a coordinate frame rotated counter-clockwise by 90 degrees. Fix this.
    q = in_msg.orientation
    print q
    r, p, y = euler_from_quaternion((q.x, q.y, q.z, q.w))
    y += math.pi / 2
    rotated = quaternion_from_euler(r, p, y)
    in_msg.orientation.x = rotated[0]
    in_msg.orientation.y = rotated[1]
    in_msg.orientation.z = rotated[2]
    in_msg.orientation.w = rotated[3]

    imu_publisher.publish(in_msg)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)  # so that control-c works
    rospy.init_node('magnetometer_republisher')

    mag_subscriber = rospy.Subscriber('~vect3_msg', Vector3Stamped_msg, mag_callback, queue_size = 2)
    mag_publisher = rospy.Publisher('~mag_msg', Mag_msg, queue_size = 2)

    imu_subscriber = rospy.Subscriber('~imu_msg_in', Imu_msg, imu_callback, queue_size = 2)
    imu_publisher = rospy.Publisher('~imu_msg_out', Imu_msg, queue_size = 2)

    rospy.spin()
