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
	RIGHT_STICK_Y_INDEX = 4;


	def __init__(self):
		rospy.init_node('joycontrol')

		self.left_pub = rospy.Publisher('left', Vector3, queue_size = 1);
		self.right_pub = rospy.Publisher('right', Vector3, queue_size = 1);

		rospy.Subscriber("joy", Joy, self.callback)

		while not rospy.is_shutdown():
			self.left_pub.publish(Vector3(self.left_y, self.left_y, 0));
			self.right_pub.publish(Vector3(self.right_y, self.right_y, 0));

	def callback(self, data):
		l_mult = None
		if data.axes[self.LEFT_STICK_Y_INDEX] <= 0:
			l_mult = 64
		else:
			l_mult = 63

		r_mult = None
		if data.axes[self.RIGHT_STICK_Y_INDEX] <= 0:
			r_mult = 64
		else:
			r_mult = 63
		self.left_y = (1 + data.axes[self.LEFT_STICK_Y_INDEX]) * l_mult
		self.right_y = (1 + data.axes[self.RIGHT_STICK_Y_INDEX]) * r_mult
		
if __name__ == '__main__':
	joy = joycontrol()
