#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Vector3


stop = 64.0

k1 = 1.0
k2 = -1.0
k3 = 1.0
k4 = -1.0
k5 = 1.0
k6 = -1.0
kt = 1.0
pubf = rospy.Publisher('front', Vector3, queue_size=1)
pubm = rospy.Publisher('mid', Vector3, queue_size=1)
pubb = rospy.Publisher('back', Vector3, queue_size=1)

def callback(data):
    x = int(data.x)
    y = int(data.y)
    rospy.loginfo("recieved: forward %i, clockwise: %i"%(x,y))
    
    fr = stop + k1*data.x +kt*data.y
    fl = stop + k2*data.x +kt*data.y
    mr = stop + k3*data.x +kt*data.y
    ml = stop + k4*data.x +kt*data.y
    br = stop + k5*data.x +kt*data.y
    bl = stop + k6*data.x +kt*data.y
    
    Front = Vector3(fr,fl,0)
    Mid = Vector3(mr,ml,0)
    Back = Vector3(br,bl,0)
    
    pubf.publish(Front)
    pubm.publish(Mid)
    pubb.publish(Back)


def driveControl():
    rospy.init_node('DriveController', anonymous=True)

    rospy.Subscriber('drive', Vector3, callback)
    
    

    

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    driveControl()
