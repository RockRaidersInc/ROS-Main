#!/usr/bin/env python
"""
This node converts joystick messages to twist messages
"""


import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy
from std_msgs.msg import Int8


class direct_drive:
    left_x = 0
    left_y = 0
    right_x = 0
    right_y = 0

    LEFT_STICK_X_INDEX = 0
    LEFT_STICK_Y_INDEX = 1
    LEFT_BUMPER_INDEX = 4
    RIGHT_STICK_X_INDEX = 3
    RIGHT_STICK_Y_INDEX = 4
    RIGHT_BUMPER_INDEX = 5
    A_BUTTON_INDEX = 0
    B_BUTTON_INDEX = 1
    X_BUTTON_INDEX = 2
    Y_BUTTON_INDEX = 3

    MAX_LINEAR_SPEED = 0.5
    MAX_ANGULAR_SPEED = 0.5
    TURBO_MAX_LINEAR_SPEED = 1.5
    TURBO_MAX_ANGULAR_SPEED = 1.5

    
    JOYSTICK_MAX_READING = 900  # any values above this wil be mapped to full speed
    JOYSTICK_DEADBAND = 0.1
    DRIVE_MAX_SPEED = 127
    DRIVE_MIN_SPEED = 0



    def __init__(self):
        rospy.init_node('joy_to_twist')
        self.left_pub = rospy.Publisher('/motors/left_pwm', Int8, queue_size=1)
        self.right_pub = rospy.Publisher('/motors/right_pwm', Int8, queue_size=1)
        rospy.Subscriber("joy", Joy, self.callback)
        # self.publish_timer = rospy.Timer(rospy.Duration(0.05), self.publish_stuff)


    def callback(self, data):
        self.left_y = data.axes[self.LEFT_STICK_Y_INDEX]
        self.left_x = data.axes[self.LEFT_STICK_X_INDEX]
        self.right_y = data.axes[self.RIGHT_STICK_Y_INDEX]
        self.right_x = data.axes[self.RIGHT_STICK_X_INDEX]
        self.bumper_l = True if data.buttons[self.LEFT_BUMPER_INDEX] == 1 else False
        self.bumper_r = True if data.buttons[self.RIGHT_BUMPER_INDEX] == 1 else False
        
        self.button_a = True if data.buttons[self.A_BUTTON_INDEX] == 1 else False
        self.button_b = True if data.buttons[self.B_BUTTON_INDEX] == 1 else False
        self.button_x = True if data.buttons[self.X_BUTTON_INDEX] == 1 else False
        self.button_y = True if data.buttons[self.Y_BUTTON_INDEX] == 1 else False

        
        def map_joystick_to_pwm(joyval):
            if abs(joyval) > self.JOYSTICK_DEADBAND:
                if joyval > 0:
                    pwm = map_to(joyval, self.JOYSTICK_DEADBAND, self.JOYSTICK_MAX_READING, 64, 127)
                    return pwm
                else:
                    pwm = map_to(joyval, self.JOYSTICK_DEADBAND, self.JOYSTICK_MAX_READING, 64, 0)
                    return pwm
            else:
                return 0
        
        left_pwm = map_joystick_to_pwm(self.left_y)
        right_pwm = map_joystick_to_pwm(self.right_y)

        print("%4i, %4i" % (left_pwm, right_pwm))

        # self.left_pub.publish(left_pwm)
        # self.right_pub.publish(right_pwm)



def map_to(x, in_low, in_high, out_low, out_high):
    val = (x - in_low) / (in_high - in_low) * (out_high - out_low) + out_low
    val = max(val, out_low)
    val = min(val, out_high)
    return val


if __name__ == '__main__':
    node = direct_drive()
    rospy.spin()
