#!/usr/bin/env python

import rospy
import socket
import numpy as np

from geometry_msgs.msg import Vector3Stamped
from sensor_msgs.msg import Joy



class ArmControl:

    RIGHT_STICK_X_INDEX = 3
    RIGHT_STICK_Y_INDEX = 4

    x_speed = 0.1
    y_speed = 0.1
    deadzone = 0.2

    def __init__(self):
        self.pose = rospy.wait_for_message('wrist_position', Vector3Stamped)

        self.arm_angles_pub = rospy.Publisher('desired_wrist_position', Vector3Stamped, queue_size = 1)
        rospy.Subscriber('joy', Joy, self.callback_joy)
        rospy.Subscriber('wrist_position', Vector3Stamped, self.callback_pose)

        while not rospy.is_shutdown():
            new_pose = self.pose
            new_pose.x += x_speed * self.right_x
            new_pose.z += y_speed * self.right_y
            rospy.sleep(1)


    def callback_joy(self, msg):
        self.right_y = msg.axes[self.RIGHT_STICK_Y_INDEX]
        self.right_x = msg.axes[self.RIGHT_STICK_X_INDEX]

        if abs(self.right_y) < self.deadzone:
            self.right_y = 0
        if abs(self.right_x) < self.deadzone:
            self.right_x = 0

    def callback_pose(self, msg):
        self.pose = msg.vector
        

if __name__ == '__main__':
    rospy.init_node('arm_control', anonymous=True)
    arm_control = ArmControl()
