#Gives the position of the end of the arm with an XYX spherical wrist end effector as a function of joint inputs

import numpy as np
import Jimbo_Kinematics.Kinematic_Utils as ku
from Jimbo_Kinematics.Actuator_Conversions import convert_to_angle

#vector from axis of elbow joint to axis of second wrist joint when in zero configuration
p_34 = ku.p_34_Spherical_Wrist
#vector from axis of second wrist joint to end of end effector when in zero configuration
p_6T = ku.p_6T_Spherical_Wrist


#forward kinematcs given 6 angles
#input: The six joint angles in radians
#output: The 4x4 transformation matrix giving the position and orientation of the end of the arm
#[ R_0T  P_0T
# 0 0 0   1  ]
#Where R_0T is the 3x3 SO(3) orientation of the end of the arm
#And P_0T is its position

def forward_kin(q1, q2, q3, q4, q5, q6):
    #Product of exponential to calculate R_0T and P_0T
    pos = ku.roty(q1).dot(ku.p_12+ku.rotx(q2).dot(ku.p_23+ku.rotx(q3).dot(p_34+ku.rotz(q4).dot(ku.rotx(q5).dot(ku.rotz(q6).dot(p_6T))))))
    orientation = ku.roty(q1).dot(ku.rotx(q2).dot(ku.rotx(q3).dot(ku.rotz(q4).dot(ku.rotx(q5).dot(ku.rotz(q6))))))
    #This stuff just puts it in the form of a transformation matrix
    result = np.concatenate((orientation, pos), axis=1)
    return np.concatenate((result, np.array([[0, 0, 0, 1]])), axis=0)
    

#forward kinematics given 4 angles and 2 linear actuator distances
#input: the four angles in radians and 2 joint distances in inches
#output: The 4x4 transformation matrix giving the position and orientation of the end of the arm
#[ R_0T  P_0T
# 0 0 0   1  ]
#Where R_0T is the 3x3 SO(3) orientation of the end of the arm
#And P_0T is its position
    
def forward_kin_distances(q1, q2, q3, q4, q5, q6):
    angles = convert_to_angle(np.array([[q2],[q3]]))
    return forward_kin(q1, angles[0][0], angles[1][0], q4, q5, q6)

