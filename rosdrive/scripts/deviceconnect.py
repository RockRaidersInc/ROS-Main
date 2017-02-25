#!/usr/bin/env python

import rospy
from std_msgs.msg import String

device1 = "ttyAMC0"
device2 = "ttyAMC1"
device3 = "ttyAMC2"


def main():
	rospy.init_node('deviceconnect', anonymous=True)
	
	pub1 = rospy.Publisher("frontdevice", String, queue_size = 1)
	pub2 = rospy.Publisher("middevice", String, queue_size = 1)
	pub3 = rospy.Publisher("backdevice", String, queue_size = 1)

	wait = rospy.Rate(1)
	x = 0
	while x < 5:
		wait.sleep()

	pub1.publish("/dev/"+device1)
	pub2.publish("/dev/"+device2)
	pub3.publish("/dev/"+device3)

	rospy.spin()




if __name__ == "__main__":
	main()
