#!/usr/bin/env python

import rospy
import roboclaw
from geometry_msgs.msg import Vector3
from std_msgs.msg import String

address = 0x80 
device = "/dev/ttyACM0"


    
def callback(data):
    rospy.loginfo('I heard %f, %f'% (data.x,data.y))

    x = int(data.x)
    y = int(data.y)
    rospy.loginfo("(x,y) = (%i,%i)"%(x,y))
    
    try:
    	roboclaw.ForwardBackwardM1(address, x)
    	roboclaw.ForwardBackwardM2(address, y)

    except:
	rospy.loginfo("cannot update motors, possible disconnect")
	connect()


def setdevice(data):
    device = "/dev"+str(data.data)
    rospy.loginfo("device set to: "+device)
    connect()

def connect():
	connect = False
        wait = rospy.Rate(1)
	while (connect==False):
    		try:
    			roboclaw.Open(device,115200)
       		 	connect = True
    		except:
			rospy.loginfo("not connecting")
		wait.sleep()

def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('motornode', anonymous=True)

    rospy.Subscriber('motor', Vector3, callback)
    rospy.Subscriber('device', String, setdevice)
    
    print("start finished")
    

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
