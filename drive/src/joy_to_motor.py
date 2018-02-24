#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Vector3 
from dependent_messages.msg import Twist2D
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

        rospy.init_node('joycontrol')

        self.back_left_pub = rospy.Publisher('front_left_motor_power', Int8, queue_size=1)
        self.back_right_pub = rospy.Publisher('front_right_motor_power', Int8, queue_size=1)
        self.front_left_pub = rospy.Publisher('back_left_motor_power', Int8, queue_size=1)
        self.front_right_pub = rospy.Publisher('back_right_motor_power', Int8, queue_size=1)

        # rospy.Subscriber('left_joystick_in', Vector3, self.left_callback)
        # rospy.Subscriber('right_joystick_in', Vector3, self.right_callback)
        rospy.Subscriber('twist_in', Twist2D, self.twist_callback)

        self.publish_timer = rospy.Timer(rospy.Duration(0.05), self.publish_stuff)  # publish joystick angles every 0.05 seconds

    def publish_stuff(self, event):
        self.front_left_pub.publish(int(self.left))
        self.front_right_pub.publish(int(self.right))
        self.back_left_pub.publish(int(self.left))
        self.back_right_pub.publish(int(self.right))

    def twist_callback(self, data):
        self.left, self.right = kinematics.inverse_kinematics(data.v, data.omega)

    def left_callback(self, data):
        self.left = map_to(data.x, self.joy_min, self.joy_max, self.motor_min, self.motor_max)

    def right_callback(self, data):
        self.right = map_to(data.x, self.joy_min, self.joy_max, self.motor_min, self.motor_max)


def map_to(x, in_min, in_max, out_min, out_max):
    """shamelessly coppied from arduino library"""
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


if __name__ == '__main__':
    joy = joycontrol()
    rospy.spin()
