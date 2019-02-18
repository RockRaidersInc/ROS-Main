#!/usr/bin/env python

import rospy
import socket
import numpy as np

from std_msgs.msg import Int16
from geometry_msgs.msg import Vector3Stamped
from Jimbo_Kinematics import Inverse_Kin_No_EE as ik



class ArmIK:

    def __init__(self):
        self.arm_angles_pub = rospy.Publisher('arm_angles', Vector3Stamped, queue_size = 1)

        self.target_pos_sub = rospy.Subscriber('wrist_position', Vector3Stamped, self.callback_target)

        while not rospy.is_shutdown():
            rospy.sleep(1)

        
    def callback_target(self, msg):
        target_pos = np.empty((3,1))
        target_pos[0][0] = msg.vector.x
        target_pos[1][0] = msg.vector.y
        target_pos[2][0] = msg.vector.z

        print(target_pos)
        print(ik.inverse_kin(target_pos))

        angles = ik.inverse_kin(target_pos)

        vec = Vector3Stamped()
        vec.vector.x = angles[0][0]
        vec.vector.y = angles[1][0]
        vec.vector.z = angles[0][0]

        self.arm_angles_pub.publish(vec)

        

if __name__ == '__main__':
    rospy.init_node('arm_ik', anonymous=True)
    arm_ik = ArmIK()
