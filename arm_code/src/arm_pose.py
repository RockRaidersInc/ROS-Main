#!/usr/bin/env python

import rospy
import socket
import numpy as np

from std_msgs.msg import Int32
from geometry_msgs.msg import Vector3Stamped
from Jimbo_Kinematics import Forward_Kin_No_EE as fk
from Jimbo_Kinematics import Actuator_Conversions as ac


class ArmPose:

    self.shoulder_enc = None
    self.elbow_enc = None
    self.turret_enc = None

    def __init__(self):
        self.arm_angles_pub = rospy.Publisher('wrist_position', Vector3Stamped, queue_size = 1)
        self.turret_enc_sub = rospy.Subscriber('turret_enc', Int32, self.callback_turret)
        self.shoulder_enc_sub = rospy.Subscriber('shoulder_enc', Int32, self.callback_shoulder)
        self.elbow_enc_sub = rospy.Subscriber('elbow_enc', Int32, self.callback_elbow)

        while not rospy.is_shutdown():
            if self.shoulder_enc is not None and self.elbow_enc is not None and self.turret_enc is not None:
                #Todo: Get turret angle
                turret_angle = self.turret_enc
                shoulder_dist = ac.shoulder_enc_to_dist(self.shoulder_enc)
                elbow_dist = ac.elbow_enc_to_dist(self.elbow_enc)

                pose = fk.forward_kin_distances(turret_angle, shoulder_dist, elbow_dist)
                position = Vector3Stamped()
                position.vector.x = pose[0][3]
                position.vector.y = pose[1][3]
                position.vector.z = pose[2][3]
                self.arm_angles_pub.publish(position)

                
            #rospy.sleep(1)

    def callback_turret(self, msg):
        self.turret_enc = msg.data

    def callback_shoulder(self, msg):
        self.shoulder_enc = msg.data
    
    def callback_elbow(self, msg):
        self.elbow_enc = msg.data     


if __name__ == '__main__':
    rospy.init_node('arm_pose', anonymous=True)
    arm_pose = ArmPose()
