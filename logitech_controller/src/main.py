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

from std_msgs.msg import String


# publish controller info
def publisher():
    pub = rospy.Publisher("logitech_controls", String, queue_size=10)
    rospy.init_node("logitech_controller", anonymous=True)
    rate = rospy.Rate(200) # 200hz cycle

    while not rospy.is_shutdown():
        # process controller input
        inputinfo = joystick_read.wait_for_changes()
        rospy.loginfo(inputinfo)
	print(inputinfo)
        pub.publish(str(inputinfo[0]) + " " + str(inputinfo[1]))
        rate.sleep()

if __name__ == "__main__":
    for i in range(0, 19):
        rospy.loginfo(joystick_read.wait_for_changes())
    try:
        publisher()
    except rospy.ROSInterruptException:
        # This should never happen, and if it does, we need to reconnect
        pass
