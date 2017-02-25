#!/usr/bin/python

#File Name main.py
#Authors Matthew Raneri

'''
Publishes input from the logitech controller (should be device js2)
'''
'''
How do you call this node?
rosrun logitech_controller main.py
'''

# Topics this node is publishing to:
# logitech controller
# publishes a string with a space, first parameter is button name or axis name
# second parameter is the value it outputs

import rospy
import joystick_read

from geometry_msgs.msg import Vector3



# publish controller info
def publisher():
    pub = rospy.Publisher("Drive", Vector3, queue_size=10)
    rospy.init_node("logitech_controller", anonymous=True)
    rate = rospy.Rate(200) # 200hz cycle

    x=0
    y=0
    k=10

    while not rospy.is_shutdown():
        # process controller input
        inputinfo = joystick_read.wait_for_changes()
        rospy.loginfo(inputinfo)
	print(inputinfo)
	if inputinfo[0]=='y':
		x=float(inputinfo[1])*-k
	elif inputinfo[0]=='x':
		y = float(inputinfo[1])*k
	pub.publish(Vector3(x,y,0))
        rate.sleep()

if __name__ == "__main__":
    for i in range(0, 19):
        rospy.loginfo(joystick_read.wait_for_changes())
    try:
        publisher()
    except rospy.ROSInterruptException:
        # This should never happen, and if it does, we need to reconnect
        pass
