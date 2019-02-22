#!/usr/bin/env python

import rospy
import socket
import numpy as np

from geometry_msgs.msg import Vector3Stamped
from sensor_msgs.msg import Joy
from std_msgs.msg import Int32



class ManualArmControl:

    RIGHT_STICK_X_INDEX = 3
    RIGHT_STICK_Y_INDEX = 4

    LEFT_BUTTON_INDEX = 4
    RIGHT_BUTTON_INDEX = 5

    SHOULDER_INITIAL_POS = 512
    ELBOW_INITIAL_POS = 512
    TURRET_STOPPED = 64
    TURRET_LEFT_SPEED = 60
    TURRET_RIGHT_SPEED = 72

    SHOULDER_MIN = 400
    SHOULDER_MAX = 600

    ELBOW_MIN = 400
    ELBOW_MAX = 600

    deadzone = 0.1

    shoulder_speed = 1
    elbow_speed = 1
    turret_speed = 1

    shoulder_dir = 0
    elbow_dir = 0

    def __init__(self):

        self.shoulder_pub = rospy.Publisher('shoulder_pos', Int32, queue_size=1)
        self.elbow_pub = rospy.Publisher('elbow_pos', Int32, queue_size=1)
        self.turret_pub = rospy.Publisher('turret_pwm', Int32, queue_size=1)

        rospy.Subscriber('joy', Joy, self.callback_joy)

        self.shoulder_pos = rospy.wait_for_message('shoulder_enc', Int32)
        self.elbow_pos = rospy.wait_for_message('elbow_enc2', Int32)
        self.turret_pwm = self.TURRET_STOPPED

        while not rospy.is_shutdown():

            self.shoulder_pos += self.shoulder_dir * self.shoulder_speed;
            if self.shoulder_pos > self.SHOULDER_MAX:
                self.shoulder_pos = self.SHOULDER_MAX
            elif self.shoulder_pos < self.SHOULDER_MIN:
                self.shoulder_pos = self.SHOULDER_MIN

            self.elbow_pos += self.elbow_dir * self.elbow_speed;
            if self.elbow_pos > self.ELBOW_MAX:
                self.elbow_pos = self.ELBOW_MAX
            elif self.elbow_pos < self.ELBOW_MIN:
                self.elbow_pos = self.ELBOW_MIN


            self.shoulder_pub.publish(self.shoulder_pos)
            self.elbow_pub.publish(self.elbow_pos)
            self.turret_pub.publish(self.turret_pwm)

            rospy.sleep(.05)


    def callback_joy(self, msg):
        right_y = msg.axes[self.RIGHT_STICK_Y_INDEX]
        right_x = msg.axes[self.RIGHT_STICK_X_INDEX]

        if abs(right_y) < self.deadzone:
            self.shoulder_dir = 0
        else:
            self.shoulder_dir = right_y / abs(right_y)
        if abs(right_x) < self.deadzone:
            self.elbow_dir = 0
        else:
            self.elbow_dir = right_x / abs(right_x)

        pass
        
        l_button = msg.buttons[self.LEFT_BUTTON_INDEX]
        r_button = msg.buttons[self.RIGHT_BUTTON_INDEX]

        if l_button:
            self.turret_pwm = self.TURRET_LEFT_SPEED
        elif r_button:
            self.turret_pwm = self.TURRET_RIGHT_SPEED
        else:
            self.turret_pwm = self.TURRET_STOPPED




    def callback_pose(self, msg):
        self.pose = msg.vector
        

if __name__ == '__main__':
    rospy.init_node('arm_control', anonymous=True)
    manual_arm_control = ManualArmControl()
