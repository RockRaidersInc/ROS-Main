#!/usr/bin/python
import rospy
import joystick_read

from std_msgs.msg import String

def publisher():
	pub = rospy.Publisher("logitech_controls", String, queue_size=10)
	rospy.init_node("logitech_controller", anonymous=True)
	rate = rospy.Rate(200) # 200hz cycle
	while not rospy.is_shutdown():
		# process controller input
		inputinfo = joystick_read.wait_for_changes()
		rospy.loginfo(inputinfo)
		pub.publish(inputinfo[0] + ":" + str(inputinfo[1]))
		rate.sleep()

if __name__ == "__main__":
	for i in range(0, 19):
		print(joystick_read.wait_for_changes())
	try:
		publisher()
	except rospy.ROSInterruptException:
		# print("Hi")
		pass
