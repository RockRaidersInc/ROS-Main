#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from std_msgs.msg import String


class joy_data:
    
    def __init__(self):
        self.joy_y1 = 0
        self.joy_y2 = 0
    
    # This pulls the data back into the joy_data class
    def callback(self, data):
        control = str(data)
        control = control.split(" ")
        if control[1] == "y":
            self.joy_y1 = float(control[2])
        elif control[1] == "ry":
            self.joy_y2 = float(control[2])


if __name__ == "__main__":
    global joy_y1, joy_y2
    jd = joy_data()
    rospy.init_node("joy2motor", anonymous=True)
    # Set up motor callback
    rospy.Subscriber("logitech_controls", String, jd.callback)
    publisher_M1 = rospy.Publisher("M1", Int32, queue_size=10)
    publisher_M2 = rospy.Publisher("M2", Int32, queue_size=10)
    rate = rospy.Rate(20) #20Hz
    old_y1 = 0
    old_y2 = 0
    while not rospy.is_shutdown():
        if old_y1 != jd.joy_y1:
            publisher_M1.publish(round(jd.joy_y1*63)+64)
            old_y1 = jd.joy_y1
        if old_y2 != jd.joy_y2:
            publisher_M2.publish(round(jd.joy_y2*63)+64)
            old_y2 = jd.joy_y2
	    rate.sleep()
