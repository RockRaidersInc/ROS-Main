#!/usr/bin/env python

# File Name (actuator_controller.py)
# Authors (Patrick Love)

# This is the general script for providing closed loop proportional
# position control to arm joints.  Instantiated for each of the 
# necessary joints


import rospy
import arm_geometry
from std_msgs.msg import Float64, Int16



# Class implementing an arbitrary joint controller with configurable encoder
# and drive topics.  Setpoints are provided by a TargetProvider

# If using PIDMotorJoint, set encoder_topic to None and prop to 1 (or a scale
# factor from angle to PID setup input if you'd like).  Then raw target angles
# will be published on the drive channel for use in setting up the PID
class JointController:
    def __init__(self, joint, encoder_topic, drive_topic, arm_topic, joint_index, prop):
        self.joint = joint
        self.arm_topic = arm_topic
        self.joint_index = joint_index
        self.encoder_topic = encoder_topic
        self.drive_topic = drive_topic
        self.target_pos = self.joint.RelaxedPos()
        self.prop = prop
    def OnNewTarget(self, arm_pose):
        self.target_pos = self.joint.EncoderForAngle(arm_pose.angles[self.joint_index])
        self.UpdatePower()
    def OnNewPosition(self, encode_msg):
        self.position = encode_msg.data
        self.UpdatePower()        
    def UpdatePower(self):
        power = self.prop * self.joint.EncoderError(self.position, self.target_pos)
        self.drive_pub.publish(power)
    def Execute(self, name):
        rospy.init_node('joint_controller_'+name)
        if self.encoder_topic is not None:
            rospy.Subscriber(self.encoder_topic, Int16, self.OnNewPosition)
        self.drive_pub = rospy.Publisher(self.drive_topic, Float64, queue_size=10)
        rospy.spin()
    