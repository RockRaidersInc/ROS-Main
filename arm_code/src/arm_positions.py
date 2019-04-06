#!/usr/bin/env python

import rospy
import socket
import numpy as np

from std_msgs.msg import Float64
from std_msgs.msg import Int32
from geometry_msgs.msg import Vector3Stamped
from Jimbo_Kinematics import Actuator_Conversions as ac

class ArmPositions:
    SHOULDER_MAX_EXTENSION = 10
    SHOULDER_MIN_EXTENSION = 0
    SHOULDER_MAX_READING = 1024
    SHOULDER_MIN_READING = 0

    ELBOW_MAX_EXTENSION = 10
    ELBOW_MIN_EXTENSION = 0
    ELBOW_MAX_READING = 1024
    ELBOW_MIN_READING = 0

    def __init__(self):
        arm_angles_sub = rospy.Subscriber('arm_angles', Vector3Stamped, self.callback_angles)

        turret_pos_pub = rospy.Publisher('turret_pos', Int32, queue_size = 1)
        shoulder_pos_pub = rospy.Publisher('shoulder_pos', Int32, queue_size = 1)
        elbow_pos_pub = rospy.Publisher('elbow_pos', Int32, queue_size = 1)


        while not rospy.is_shutdown():
            rospy.sleep(1)

        

    def callback_angles(self, msg):
        target_angles = np.array([[msg.vector.y],[msg.vector.z]])
        print(target_angles)

        target_distances = ac.convert_to_distance(target_angles)
        print(target_distances)
        shoulder_enc = ac.shoulder_dist_to_enc(target_distances[0][0])
        elbow_enc = ac.elbow_dist_to_enc(target_distances[1][0])
        target_positions = np.array([[shoulder_enc],[elbow_enc]])
        print(target_positions)

        if not np.isnan(target_positions).any():
            turret_pos = Int32()
            shoulder_pos = Int32()
            elbow_pos = Int32()

            # Populate variabiles

            turret_pos_pub.publish(turret_pos)
            shoulder_pos_pub.publish(shoulder_pos)
            elbow_pos_pub.publish(elbow_pos)



if __name__ == '__main__':
    rospy.init_node('arm_positions', anonymous=True)
    arm_positions = ArmPositions()
