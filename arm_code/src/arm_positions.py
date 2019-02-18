#!/usr/bin/env python

import rospy
import socket
import numpy as np

from std_msgs.msg import Float64
from std_msgs.msg import Int32
from geometry_msgs.msg import Vector3Stamped
from Jimbo_Kinematics import Actuator_Conversions as ac

class ArmIK:
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
        target_angles = np.empty((2, 1))
        target_angles[0][0] = msg.vector.y;
        target_angles[1][0] = msg.vector.z;

        target_distances = ac.convert_to_distance(target_angles)
        print(target_distances)
        target_positions = self.convert_to_positions(target_distances)
        print(target_positions)

    def convert_to_positions(self, distances):
        p1 = (distances[0][0] - self.SHOULDER_MIN_EXTENSION) * (self.SHOULDER_MAX_READING - self.SHOULDER_MIN_READING) / (self.SHOULDER_MAX_EXTENSION - self.SHOULDER_MIN_EXTENSION) + self.SHOULDER_MIN_READING
        p2 = (distances[1][0] - self.ELBOW_MIN_EXTENSION) * (self.ELBOW_MAX_READING - self.ELBOW_MIN_READING) / (self.ELBOW_MAX_EXTENSION - self.ELBOW_MIN_EXTENSION) + self.ELBOW_MIN_READING
        return np.array([[p1],[p2]])
        

if __name__ == '__main__':
    rospy.init_node('arm_ik', anonymous=True)
    arm_ik = ArmIK()
