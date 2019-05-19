#!/usr/bin/env python
"""
This node lets a user stop the autonomy system from publishing motor commands by moving a joystick.
Pressing a gives control back to the autonomy system. This node is implemented by splitting cmd_vel
up into two topics, a topic for joysticks to publish to (~cmd_vel_user) and a topic for the
navstack to publish to (~cmd_vel_autonomy). It also directly subscribes to the joystick topic so
that it can tell when the a button is pressed.
"""


import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy


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


class CmdVelMultiplexer:
    def __init__(self):
        rospy.init_node('cmd_vel_multiplexer')    
        self.user_in_controll = False
        self.twist_pub = rospy.Publisher('~cmd_vel_out', Twist, queue_size=1)
        rospy.Subscriber('~cmd_vel_user', Twist, self.user_callback)
        rospy.Subscriber('~cmd_vel_autonomy', Twist, self.autonomy_callback)
        rospy.Subscriber('~joy', Joy, self.joy_callback)

    def autonomy_callback(self, data):
        if not self.user_in_controll:
            self.twist_pub.publish(data)

    def user_callback(self, data):
        if abs(data.linear.x) > 0.0001 or abs(data.angular.z) > 0.0001:
            if not self.user_in_controll:
                rospy.loginfo("controll taken away from autonomy system")
            self.user_in_controll = True
        self.twist_pub.publish(data)

    def joy_callback(self, data):
        if data.buttons[A_BUTTON_INDEX] == 1:
            self.user_in_controll = False
            rospy.loginfo("controll given back to autonomy system")


if __name__ == '__main__':
    node = CmdVelMultiplexer()
    rospy.spin()
