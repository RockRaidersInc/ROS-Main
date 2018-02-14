#!/usr/bin/env python

import rospy
import roboclaw
from geometry_msgs.msg import Vector3
from std_msgs.msg import String
from sensor_msgs.msg import Joy

class joycontrol:
	left_pub = None
	right_pub = None

	left_y = 0
	right_y = 0

	LEFT_STICK_Y_INDEX = 1;
	RIGHT_STICK_Y_INDEX = 3;


	def __init__(self, n):
		self.left_pub = rospy.Publisher('left', Vector3, queue_size = 1);
		self.right_pub = rospy.Publisher('right', Vector3, queue_size = 1);

		rospy.Subscriber("joy", Joy, self.callback)

		while not rospy.is_shutdown():
			self.left_pub.publish(Vector3(left_y, left_y, 0));
			self.right_pub.publish(Vector3(right_y, right_y, 0));

	def callback(self, data):
		left_y = (1 + data.axes[LEFT_STICK_Y_INDEX]) * 64;
		right_y = (1 + data.axes[RIGHT_STICK_Y_INDEX]) * 64;
		
