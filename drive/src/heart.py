#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Vector3


class heart:
	x = 0
	y = 0
	z = 0
	pub = None
	def update(self,data):
		self.x = data.x
		self.y = data.y
		self.z = data.z
	

	def __init__(self):
		rospy.init_node("heartbeat", anonymous = True)
	
		self.pub = rospy.Publisher("drive", Vector3, queue_size = 1)
		rospy.Subscriber("beat", Vector3, self.update)
		self.loop()

	def loop(self):
		wait = rospy.Rate(50)
		while not rospy.is_shutdown():
			print("x,y,z = %i,%i,%i"%(self.x,self.y,self.z))
			self.pub.publish(Vector3(self.x,self.y,self.z))
			wait.sleep()

		


if __name__ == "__main__":
	heart()
