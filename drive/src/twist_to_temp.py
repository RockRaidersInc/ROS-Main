#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Vector3, Twist
# from dependent_messages.msg import Twist2D
from std_msgs.msg import Int8
import kinematics

class joycontrol:
    def __init__(self):
        self.left = 0
        self.right = 0

        self.motor_min = 64
        self.motor_max = -64

        self.joy_max = 126
        self.joy_min = 0

        rospy.init_node('joycontrol_temp')
        self.left_pub = rospy.Publisher('left', Vector3, queue_size = 1)
        self.right_pub = rospy.Publisher('right', Vector3, queue_size = 1)

        # rospy.Subscriber('left_joystick_in', Vector3, self.left_callback)
        # rospy.Subscriber('right_joystick_in', Vector3, self.right_callback)
        rospy.Subscriber('/twist', Twist, self.twist_callback)

        self.publish_timer = rospy.Timer(rospy.Duration(0.2), self.publish_stuff)  # publish joystick angles every 0.05 seconds


    def publish_stuff(self, event):
        self.left_pub.publish(Vector3(self.left, self.left, 0))
        self.right_pub.publish(Vector3(self.right, self.right, 0))

    def twist_callback(self, data):
        self.left, self.right = kinematics.inverse_kinematics(max(min(data.linear.x, 64), -64), max(min(data.angular.z, 64), -64))


def map_to(x, in_min, in_max, out_min, out_max):
    """shamelessly coppied from arduino library"""
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


if __name__ == '__main__':
    joy = joycontrol()
    rospy.spin()
