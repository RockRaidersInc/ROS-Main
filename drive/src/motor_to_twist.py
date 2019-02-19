#!/usr/bin/env python

import rospy
from std_msgs.msg import Int32
import std_msgs.msg
from nav_msgs.msg import Odometry

import kinematics
import numpy as np


class motor_to_twist:
    """
    This class takes in measured motor data (actual wheel movement) and publishes twists as odometry messages.
    This data is used by the localization nodes.
    """

    def __init__(self):
        self.left_positions = list()
        self.right_positions = list()

        self.track = rospy.get_param("/rover_constants/wheel_base_width")
        self.wheel_diameter = rospy.get_param("/rover_constants/wheel_diameter")
        self.encoder_ticks_per_rad = rospy.get_param("/rover_constants/encoder_ticks_per_rad")

        rospy.Subscriber("left_motor_in", Int32, self.left_callback)
        rospy.Subscriber("right_motor_in", Int32, self.right_callback)
        self.pub = rospy.Publisher("twist_publisher", Odometry, queue_size = 1)
        self.publish_timer = rospy.Timer(rospy.Duration(0.05), self.publish_data)

    def left_callback(self, data):
        new_pos = data.data / self.encoder_ticks_per_rad
        self.left_positions.append((new_pos, rospy.Time.now()))

    def right_callback(self, data):
        new_pos = data.data / self.encoder_ticks_per_rad
        self.right_positions.append((new_pos, rospy.Time.now()))
    
    def publish_data(self, time_obj):
        while len(self.left_positions) > 2:
            self.left_positions.pop(0)
        while len(self.right_positions) > 2:
            self.right_positions.pop(0)

        out_msg = Odometry()
        out_msg.header.stamp = rospy.Time.now()
        out_msg.child_frame_id = "base_link"

        try:
            if len(self.left_positions) == 2 and len(self.right_positions) == 2:
                left_vel = (self.left_positions[1][0] - self.left_positions[0][0]) / (self.left_positions[1][1] - self.left_positions[0][1]).to_sec()
                right_vel = (self.right_positions[1][0] - self.right_positions[0][0]) / (self.right_positions[1][1] - self.right_positions[0][1]).to_sec()
                v, omega = kinematics.forward_kinematics(left_vel, right_vel, track=self.track, diameter=self.wheel_diameter)
                out_msg.twist.twist.linear.x = v
                out_msg.twist.twist.angular.z = omega
                self.pub.publish(out_msg)

                # don't use old data
                self.left_positions.pop(0)
                self.right_positions.pop(0)
            
            else:
                # if no data is being recieved then the motor control node is probably not running. Publish zero velocity.
                out_msg.twist.twist.linear.x = 0
                out_msg.twist.twist.angular.z = 0
                self.pub.publish(out_msg)
        except:
            pass


if __name__ == '__main__':
    rospy.init_node('motor_to_twist')
    controller = motor_to_twist()
    rospy.spin()
