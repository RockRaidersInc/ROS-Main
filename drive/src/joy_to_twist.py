#!/usr/bin/env python
"""
This node converts joystick messages to twist messages
"""


import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy


class joycontrol:
    left_x = 0
    left_y = 0
    right_x = 0
    right_y = 0

    # definitions of which buttons on the joystick match to which index in the message
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

    MAX_LINEAR_SPEED = 1.0
    MAX_ANGULAR_SPEED = 0.75
    TURBO_MAX_LINEAR_SPEED = 1.5
    TURBO_MAX_ANGULAR_SPEED = 0.75

    THRESHOLD = 0.1

    def __init__(self):
        rospy.init_node('joy_to_twist')
        self.twist_pub = rospy.Publisher('~cmd_vel', Twist, queue_size=1)
        rospy.Subscriber("joy", Joy, self.callback)
        # self.publish_timer = rospy.Timer(rospy.Duration(0.05), self.publish_stuff)

    def callback(self, data):
        """
        Record data from the incoming joystick message then call the function to publish a twist message
        """
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

        self.publish_stuff(None)

    def publish_stuff(self, event):
        """
        Take saved joystick positions and publish a twist message. Basically joystick up/down is 
        directly converted into linear motion and joystick left/right is directly converted to angular motion.
        """
        twist_msg = Twist()

        # Joysticks don't always return to exactly zero when let go of. The threshold is here so that if it's really
        # close to zero then teh software will assume it's supposed to be zero. Otherwise the rover will try to creep
        # forward really slowly.
        if abs(self.left_y) > self.THRESHOLD or abs(self.left_x) > self.THRESHOLD:

            # x is the "turbo" button. The rover drives much faster when it's pressed (so that taking it places is faster)
            if self.button_x:
                twist_msg.linear.x = self.left_y * self.TURBO_MAX_LINEAR_SPEED
                twist_msg.angular.z = self.left_x * self.TURBO_MAX_ANGULAR_SPEED
            else:
                twist_msg.linear.x = self.left_y * self.MAX_LINEAR_SPEED
                twist_msg.angular.z = self.left_x * self.MAX_ANGULAR_SPEED
        else:
            twist_msg.linear.x = 0.0
            twist_msg.angular.z = 0.0
        self.twist_pub.publish(twist_msg)


if __name__ == '__main__':
    joy = joycontrol()

    # This function forever. It's needed because otherwise the node wil exit.
    rospy.spin()
