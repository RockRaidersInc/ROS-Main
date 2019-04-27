#Finds the joint angles needed to move the end of the arm (with no end effector) to the given position
#The end is defined as the point at the end of the last segment,
#centered in the x-y cross section when in the zero configuration

import numpy as np
import Jimbo_Kinematics.Kinematic_Utils as ku

#vector from axis of elbow joint to end of second link when in zero configuration
p_3T = ku.p_3T_No_EE


#Inverse kinematics given the desired position
#input: 1x3 vector (numpy array) representing the desired position
#output: 1x3 vector (numpy array) containing the three joint angles
#If any component of the output is Nan, the solution can't be reached

def inverse_kin(p_0T):
    #Solving for q1
    #Assumes that arm will not reach behind the central axis
    q1  = np.arctan2(p_0T[0][0], p_0T[2][0])
    q3_sols = ku.subproblem3(p_3T, -1*ku.p_23, np.linalg.norm(ku.roty(-1*q1).dot(p_0T)-ku.p_12))
    #Two solutions to this problem
    first_q3 = q3_sols[0]
    second_q3 = q3_sols[1]
    #The problem is already solved, I just try to eliminate solutions that I know won't work here
    q3_test = np.pi/2-ku.q3_zero-first_q3 #angle between the two segments
    if q3_test>np.pi: #Constrain the angle between pi and -pi
        q3_test-=2*np.pi
    if q3_test<-1*np.pi:
        q3_test+=2*np.pi
    if q3_test>0 and q3_test<np.pi: #if the angle isn't in this range, the joint is bending the wrong way
        first_q2 = ku.subproblem1_X(ku.p_23+ku.rotx(first_q3).dot(p_3T), ku.roty(-1*q1).dot(p_0T)-ku.p_12)
        return np.array([[q1],[first_q2],[first_q3]])
    else:
        second_q2 = ku.subproblem1_X(ku.p_23+ku.rotx(second_q3).dot(p_3T), ku.roty(-1*q1).dot(p_0T)-ku.p_12)
        return np.array([[q1],[second_q2],[second_q3]])
