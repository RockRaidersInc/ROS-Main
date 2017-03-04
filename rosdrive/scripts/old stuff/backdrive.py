#!/usr/bin/env python

import rospy
import roboclaw
from geometry_msgs.msg import Vector3
from std_msgs.msg import String

address = 0x80 
device = "/dev/ttyACM2"   
    
def callback(data):
    rospy.loginfo(rospy.get_caller_id() + 'I heard %f, %f'% (data.x,data.y))

    x = int(data.x)
    y = int(data.y)
    rospy.loginfo("(x,y) = (%i,%i)"%(x,y))
    
    roboclaw.ForwardBackwardM1(address, x)
    roboclaw.ForwardBackwardM2(address, y)

def setdevice(data):
    device = "/dev"+str(data.data)
    rospy.loginfo("device set to: "+device)

def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('mcController', anonymous=True)

    rospy.Subscriber('motor', Vector3, callback)
    rospy.Subscriber('device', String, setdevice)
    connect = False
    while !connect:
    	try:
    		roboclaw.Open(device,115200)
        	connect = True
    	except:
		print("not connecting")

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
