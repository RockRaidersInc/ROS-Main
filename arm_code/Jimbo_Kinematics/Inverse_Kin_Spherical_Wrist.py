#Finds the joint angles needed to move the end of the arm with a spherical
#wrist end effector to the given position and orientation

import numpy as np
import Jimbo_Kinematics.Kinematic_Utils as ku

#vector from axis of elbow joint to axis of second wrist joint when in zero configuration
p_34 = ku.p_34_Spherical_Wrist
#vector from axis of second wrist joint to end of end effector when in zero configuration
p_6T = ku.p_6T_Spherical_Wrist

#Some helpful vector definitions
ex = np.array([[1],[0],[0]])
ez = np.array([[0],[0],[1]])


#Inverse kinematics given the desired position and orientation
#input: 1x3 vector (numpy array) representing the desired position and
#       3x3 SO(3) matrix (numpy array) representing the desired orientation
#output: 2x6 vector (numpy array) containing two columns of six angles, each forming a solution
#If any component of a column is Nan, that solution can't be reached

def inverse_kin(p_0T, R_0T):
    #Decouple orientation to solve for position
    p_06 = p_0T - R_0T.dot(p_6T)
    #Solving for q1
    #Assumes that arm will not reach behind the central axis
    q1  = np.arctan2(p_06[0][0], p_06[2][0])
    q3_sols = ku.subproblem3(p_34, -1*ku.p_23, np.linalg.norm(ku.roty(-1*q1).dot(p_06)-ku.p_12))
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
        q2 = ku.subproblem1_X(ku.p_23+ku.rotx(first_q3).dot(p_34), ku.roty(-1*q1).dot(p_06)-ku.p_12)
        q3 = first_q3
    else:
        q2 = ku.subproblem1_X(ku.p_23+ku.rotx(second_q3).dot(p_34), ku.roty(-1*q1).dot(p_06)-ku.p_12)
        q3 = second_q3
    #Now solve for wrist angles to satisfy the orientation    
    R_3T = ku.rotx(-1*q3).dot(ku.rotx(-1*q2).dot(ku.roty(-1*q1).dot(R_0T)))
    first_sols = ku.subproblem2(ez, R_3T.dot(ez)) #gets the two possible solutions for q4 and q5
    #find q6 for each possible solution
    q6_1 = ku.subproblem1_Z(ex, ku.rotx(-1*first_sols[1][0]).dot(ku.rotz(-1*first_sols[0][0]).dot(R_3T.dot(ex))))
    q6_2 = ku.subproblem1_Z(ex, ku.rotx(-1*first_sols[1][1]).dot(ku.rotz(-1*first_sols[0][1]).dot(R_3T.dot(ex))))
    return np.array([[q1, q1],[q2, q2],[q3, q3],[first_sols[0][0], first_sols[0][1]],[first_sols[1][0], first_sols[1][1]],[q6_1, q6_2]])