#Gives the position of the end of the arm with no end effector as a function of joint inputs
#The end is defined as the point at the end of the last segment,
#centered in the x-y cross section when in the zero configuration

import numpy as np
import Jimbo_Kinematics.Kinematic_Utils as ku
from Jimbo_Kinematics.Actuator_Conversions import convert_to_angle

#vector from axis of elbow joint to end of second link when in zero configuration
p_3T = ku.p_3T_No_EE


#forward kinematics given 3 angles
#input: The three joint angles in radians
#output: The 4x4 transformation matrix giving the position and orientation of the end of the arm
#[ R_0T  P_0T
# 0 0 0   1  ]
#Where R_0T is the 3x3 SO(3) orientation of the end of the arm
#And P_0T is its position

def forward_kin(q1, q2, q3):
    #Product of exponential to calculate R_0T and P_0T
    pos = ku.roty(q1).dot(ku.p_12 + ku.rotx(q2).dot(ku.p_23 + ku.rotx(q3).dot(p_3T)))
    orientation = ku.roty(q1).dot(ku.rotx(q2).dot(ku.rotx(q3)))
    #This stuff just puts it in the form of a transformation matrix
    result = np.concatenate((orientation, pos), axis=1)
    return np.concatenate((result, np.array([[0, 0, 0, 1]])), axis=0)


#forward kinematics given the base angle and 2 linear actuator distances
#input: the base angle in radians and 2 joint distances in inches
#output: The 4x4 transformation matrix giving the position and orientation of the end of the arm
#[ R_0T  P_0T
# 0 0 0   1  ]
#Where R_0T is the 3x3 SO(3) orientation of the end of the arm
#And P_0T is its position
    
def forward_kin_distances(q1, q2, q3):
    angles = convert_to_angle(np.array([[q2],[q3]]))
    return forward_kin(q1, angles[0][0], angles[1][0])