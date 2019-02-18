#Some useful functions for arm kinematics
#Also contains useful constants regarding the dimensions of the arm
#The linear actuator zero positions are set at the end of this file

import numpy as np


#Rotation matrix for x axis rotation
#input: Angle of rotation about X axis in radians
#output: SO(3) rotation matrix (numpy array)

def rotx(q):
    return np.array([[1, 0, 0],[0, np.cos(q), -1*np.sin(q)],[0, np.sin(q), np.cos(q)]])


#Rotation matrix for y axis rotation
#input: Angle of rotation about Y axis in radians
#output: SO(3) rotation matrix (numpy array)
    
def roty(q):
    return np.array([[np.cos(q), 0, np.sin(q)],[0, 1, 0],[-1*np.sin(q), 0, np.cos(q)]])


#Rotation matrix for z axis rotation
#input: Angle of rotation about Z axis in radians
#output: SO(3) rotation matrix (numpy array)
    
def rotz(q):
    return np.array([[np.cos(q), -1*np.sin(q), 0],[np.sin(q), np.cos(q), 0],[0, 0, 1]])

#Solves the Paden-Kahan Subproblem 1 for a rotation about the X axis
#Finds angle you need to rotate vector p for it to be collinear with vector q
#input: 2 3x1 vectors (numpy arrays) as defined above    
#output: The necessary angle of rotation to solve the problem     
    
def subproblem1_X(p,q):
    return np.arctan2(p[1][0]*q[2][0]-p[2][0]*q[1][0], p[1][0]*q[1][0]+p[2][0]*q[2][0])


#Solves the Paden-Kahan Subproblem 1 for a rotation about the Z axis
#Finds angle you need to rotate vector p for it to be collinear with vector q
#input: 2 3x1 vectors (numpy arrays) as defined above    
#output: The necessary angle of rotation to solve the problem  
    
def subproblem1_Z(p,q):
    return np.arctan2(p[0][0]*q[1][0]-p[1][0]*q[0][0], p[0][0]*q[0][0]+p[1][0]*q[1][0])


#Solves the Paden-Kahan Subproblem 2
#Vector p is rotated twice (Z, then X)  until it is collinear with q
#Returns the two angles need to achieve the above
#Generally has 2 solutions
#input: 2 3x1 vectors (numpy arrays) as defined above    
#output: A 2x2 matrix (numpy array) containing the two sets of two angles of rotation 
#needed to solve the problem
#each solution is listed along a column
#[first solution angle 1    second solution angle 1
# first solution angle 2    second solution angle 2]  

def subproblem2(p,q):
    z1 = np.array([[p[0][0]],[np.sqrt(1-p[0][0]**2-q[2][0]**2)],[q[2][0]]])
    z2 = np.array([[p[0][0]],[-1*np.sqrt(1-p[0][0]**2-q[2][0]**2)],[q[2][0]]])
    return np.array([[-1*subproblem1_Z(q, z1), -1*subproblem1_Z(q, z2)], [subproblem1_X(p, z1), subproblem1_X(p, z2)]])


#Solves the Paden-Kahan Subproblem 3
#Finds intersection of a cone and a sphere
#In a plane, this simplifies to rotating vector p around the end of vector q
#Returns the angles that result in p intersecting a circle of radius d centered at the origin
#Generally returns two solutions
#input: 2 3x1 vectors p and q (numpy arrays) and a scalar d as defined above  
#output: a 1x2 row vector (numpy array) containing the two angles that solve the problem

def subproblem3(p,q,d):
    phi = np.arccos((p[1][0]**2+p[2][0]**2+q[1][0]**2+q[2][0]**2-d**2)/(2*np.sqrt((p[1][0]**2+p[2][0]**2)*(q[1][0]**2+q[2][0]**2))))
    theta = subproblem1_X(p,q)
    return np.array([theta+phi, theta-phi])


#Length of first linear actuator (attached to base) at zero position
LA2_ZERO_POSITION = 23.9387

#Length of second linear actuator (attached to second segment) at zero position
LA3_ZERO_POSITION = 19.5272

#Calculates the zero angle values from the zero positions of the actuators
q2_zero=np.pi/2-np.arctan(1/40)-np.arctan(14/15)-np.arccos((426.5625-(LA2_ZERO_POSITION)**2)/(2*np.sqrt(10531.57813)))
q3_zero=np.arctan(1/37) + np.arccos((388.0625-(LA3_ZERO_POSITION)**2)/(13.5*np.sqrt(342.5)))-np.pi/2

#ARM DIMENSIONS
#vector from origin to axis of first horizontal joint
p_12 = np.array([[0],[7.683],[-1.5]]) 
#vector from axis of horizontal shoulder joint to axis of elbow joint when in zero configuration
#p_23 = rotx(q2_zero).dot(np.array([[0],[0],[24.5]])) 
p_23 = rotx(q2_zero).dot(np.array([[0],[24.5],[0]])) 
#vector from axis of elbow joint to end of second link when in zero configuration
p_3T_No_EE = rotx(q2_zero+q3_zero+np.pi/2).dot(np.array([[0],[0],[22]])) 

#SPHERICAL WRIST DIMENSIONS
#vector from axis of elbow joint to axis of second wrist joint when in zero configuration
p_34_Spherical_Wrist = np.array([[0],[0],[22]])
#vector from axis of second wrist joint to end of end effector when in zero configuration
p_6T_Spherical_Wrist = np.array([[1],[1],[1]])
