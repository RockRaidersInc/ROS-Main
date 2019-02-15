#!/usr/bin/env python
"""
This node converts joystick messages to twist messages
"""

import rospy
import roboclaw
from geometry_msgs.msg import Vector3, Twist
# from dependent_messages.msg import Twist2D
from std_msgs.msg import String
from sensor_msgs.msg import Joy

class joycontrol:
    left_x = 0
    left_y = 0
    right_x = 0
    right_y = 0

    LEFT_STICK_X_INDEX = 0
    LEFT_STICK_Y_INDEX = 1
    RIGHT_STICK_X_INDEX = 3
    RIGHT_STICK_Y_INDEX = 4

    MAX_LINEAR_SPEED = 2
    MAX_ANGULAR_SPEED = .5


    def __init__(self):
        rospy.init_node('joycontrol')

        self.twist_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        rospy.Subscriber("joy", Joy, self.callback)

        self.publish_timer = rospy.Timer(rospy.Duration(0.05), self.publish_stuff)  # publish joystick angles every 0.05 seconds


    def publish_stuff(self, event):
        twist_msg = Twist()
        twist_msg.linear.x = self.left_y * self.MAX_LINEAR_SPEED
        twist_msg.angular.z = self.left_x * self.MAX_ANGULAR_SPEED
        self.twist_pub.publish(twist_msg)


    def callback(self, data):
        self.left_y = data.axes[self.LEFT_STICK_Y_INDEX]
        self.left_x = data.axes[self.LEFT_STICK_X_INDEX]
        self.right_y = data.axes[self.RIGHT_STICK_Y_INDEX]
        self.right_x = data.axes[self.RIGHT_STICK_X_INDEX]


        
        
if __name__ == '__main__':
    joy = joycontrol()
    rospy.spin()
