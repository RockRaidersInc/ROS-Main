#!/usr/bin/env python

"""
This node does something called foward kinematics. It reads the motor speeds and publishes a twist representing
what the rover is actually doing.
"""

import rospy
from std_msgs.msg import Int32
import std_msgs.msg
from nav_msgs.msg import Odometry

import numpy as np

import kinematics
import numpy as np


class motor_to_twist:
    """
    This class takes in measured motor data (actual wheel movement) and publishes twists as odometry messages.
    This data is used by the localization nodes.
    """

    def __init__(self, linear_cov_factor, angular_cov_factor):
        self.left_positions = list()
        self.right_positions = list()
        self.linear_cov_factor = linear_cov_factor
        self.angular_cov_factor = angular_cov_factor

        self.track = rospy.get_param("/rover_constants/wheel_base_width")
        self.wheel_diameter = rospy.get_param("/rover_constants/wheel_diameter")
        self.encoder_ticks_per_rad = rospy.get_param("/rover_constants/encoder_ticks_per_rad")

        rospy.Subscriber("left_motor_in", Int32, self.left_callback)
        rospy.Subscriber("right_motor_in", Int32, self.right_callback)
        self.pub = rospy.Publisher("twist_publisher", Odometry, queue_size = 1)
        self.publish_timer = rospy.Timer(rospy.Duration(0.1), self.publish_data)

    def left_callback(self, data):
        new_pos = data.data / self.encoder_ticks_per_rad
        self.left_positions.append((new_pos, rospy.Time.now()))

    def right_callback(self, data):
        new_pos = data.data / self.encoder_ticks_per_rad
        self.right_positions.append((new_pos, rospy.Time.now()))
    
    def publish_data(self, time_obj):
        if len(self.left_positions) <= 10 or len(self.right_positions) <= 10:
            return

        while self.left_positions[-1][1].to_sec() - self.left_positions[0][1].to_sec() > 1:
            self.left_positions.pop(0)
        while self.right_positions[-1][1].to_sec() - self.right_positions[0][1].to_sec() > 1:
            self.right_positions.pop(0)

        if len(self.left_positions) <= 10 or len(self.right_positions) <= 10:
            return

        out_msg = Odometry()
        out_msg.header.stamp = rospy.Time.now()
        out_msg.child_frame_id = "base_link"

        try:
            if len(self.left_positions) > 2 and len(self.right_positions) > 2:
                left_vel = self.linear_reg_slope(self.left_positions)
                right_vel = self.linear_reg_slope(self.right_positions)

                v, omega = kinematics.forward_kinematics(left_vel, right_vel, track=self.track, diameter=self.wheel_diameter)
                out_msg.twist.twist.linear.x = v
                out_msg.twist.twist.angular.z = omega

                # don't use old data
                self.left_positions.pop(0)
                self.right_positions.pop(0)
                # rospy.logerr("motor_to_twist setting twists: " + str(v) + " " + str(omega))
            
            else:
                # if no data is being recieved then the motor control node is probably not running. Publish zero velocity.
                out_msg.twist.twist.linear.x = 0
                out_msg.twist.twist.angular.z = 0
            
            self.set_covariance(out_msg)
            self.pub.publish(out_msg)
            # rospy.logerr("motor_to_twist published")
        except:
            pass


    def linear_reg_slope(self, data):
        """ calculates the first derivative of noisy data with a linear regression """
        time = np.array([d[1].to_sec() for d in data])
        position = np.array([d[0] for d in data])
        m, b = np.polyfit(time, position, 1)
        return m


    def set_covariance(self, msg):
        """
        Twist messages have a covarrinace. The covariance is basically a measure of possible error, very similar
        to a standard deviation. For example, if msg.twist.twist.linear.x = 0.5 and msg.twist.covariance[0] = 0.1 
        would mean that the linear velocity is 0.5 +- 0.1 m/s with a 68% confidence.
        """
        linear_factor = abs(msg.twist.twist.linear.x * self.linear_cov_factor)
        angular_factor = abs(msg.twist.twist.angular.z * self.angular_cov_factor)

        # set the x and y covariance. Set y because the rover might slip sideways if it goes over rough terrain
        msg.twist.covariance[0*6 + 0] = linear_factor
        msg.twist.covariance[1*6 + 1] = linear_factor
        msg.twist.covariance[3*6 + 5] = angular_factor



if __name__ == '__main__':
    rospy.init_node('motor_to_twist')
    linear_cov_factor = rospy.get_param('~linear_covariance_scale_factor', 0)
    angular_cov_factor = rospy.get_param('~angular_covariance_scale_factor', 0)
    controller = motor_to_twist(linear_cov_factor, angular_cov_factor)
    rospy.spin()
