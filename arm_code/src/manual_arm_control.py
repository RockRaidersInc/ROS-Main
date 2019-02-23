#!/usr/bin/env python

import rospy
import socket
import numpy as np

from geometry_msgs.msg import Vector3Stamped
from sensor_msgs.msg import Joy
from std_msgs.msg import Int8
from std_msgs.msg import Int32



class ManualArmControl:

    RIGHT_STICK_X_INDEX = 3
    RIGHT_STICK_Y_INDEX = 4

    LEFT_BUTTON_INDEX = 4
    RIGHT_BUTTON_INDEX = 5

    SHOULDER_INITIAL_POS = 512
    ELBOW_INITIAL_POS = 512
    TURRET_STOPPED = 64
    TURRET_LEFT_SPEED = 126
    TURRET_RIGHT_SPEED = 1

    # TURRET_STOPPED = 0
    # TURRET_LEFT_SPEED = -300
    # TURRET_RIGHT_SPEED = 300

    SHOULDER_MIN = 670
    SHOULDER_MAX = 1800

    ELBOW_MIN = 200
    ELBOW_MAX = 1400

    deadzone = 0.1

    shoulder_speed = 4
    elbow_speed = 10
    turret_speed = 1

    shoulder_dir = 0
    elbow_dir = 0

    def __init__(self):

        self.shoulder_pub = rospy.Publisher('/motors/shoulder_pos', Int32, queue_size=1)
        self.elbow_pub = rospy.Publisher('/motors/elbow_pos', Int32, queue_size=1)
        self.turret_pub = rospy.Publisher('/motors/turret_pwm', Int8, queue_size=1)

        rospy.Subscriber('joy', Joy, self.callback_joy)

        self.shoulder_pos = rospy.wait_for_message('/motors/shoulder_enc', Int32)
        print("self.shoulder_pos: ", self.shoulder_pos.data)
        self.elbow_pos = rospy.wait_for_message('/motors/elbow_enc', Int32)
        print("self.elbow_pos: ", self.elbow_pos.data)
        self.turret_pwm = self.TURRET_STOPPED

        while not rospy.is_shutdown():

            self.shoulder_pos.data += self.shoulder_dir * self.shoulder_speed
            if self.shoulder_pos.data > self.SHOULDER_MAX:
                self.shoulder_pos.data = self.SHOULDER_MAX
            elif self.shoulder_pos.data < self.SHOULDER_MIN:
                self.shoulder_pos.data = self.SHOULDER_MIN

            self.elbow_pos.data += self.elbow_dir * self.elbow_speed
            if self.elbow_pos.data > self.ELBOW_MAX:
                self.elbow_pos.data = self.ELBOW_MAX
            elif self.elbow_pos.data < self.ELBOW_MIN:
                self.elbow_pos.data = self.ELBOW_MIN

            print("self.shoulder_pos: ", self.shoulder_pos.data)
            print("self.elbow_pos: ", self.elbow_pos.data)
            print("self.turret_pwm: ", self.turret_pwm)

            turret_msg = Int8()
            turret_msg.data = self.turret_pwm

            self.shoulder_pub.publish(self.shoulder_pos)
            self.elbow_pub.publish(self.elbow_pos)
            self.turret_pub.publish(turret_msg)

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

        print("self.shoulder_dir: ", self.shoulder_dir)
        print("self.elbow_dir: ", self.elbow_dir)
        
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
