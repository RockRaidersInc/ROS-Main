#!/usr/bin/env python

#NEEDS TESTING!

#TODO: Eliminate the wheel that's slipping the most before taking pseudo-inverse

#Last edited by Connor McGowan 9/3/19

from math import pi, sqrt, sin, cos, atan, atan2, fabs
from numpy import array, zeros, dot, linalg

#These should go in a config file
#These values define the dimensions of the rover, as well as the location of the
#point where the velocities are measured (reference point)

#Diameter of wheel, in meters
wheel_diameter = 3

#Distance between wheels on opposite sides of the chassis, in meters
track = 1

#Distance from the reference point to the line passing through both front wheels, in meters
#If the reference point is in front of the front wheels, this number will be negative
front_wheelbase = 2

#Distance from the reference point to the line passing through both rear wheels, in meters
#If the reference point is behind the rear wheels, this number will be negative
rear_wheelbase = 3

#Establish coordinates of the center of each wheel relative to the reference point
    #Wheel definitions:
    #0 = Front Left
    #1 = Front Right
    #2 = Back Left
    #3 = Back Right
wheel_y_coords = [track/2.0, track/-2.0, track/2.0, track/-2.0]
wheel_x_coords = [front_wheelbase, front_wheelbase, -1*rear_wheelbase, -1*rear_wheelbase]

#Transform matrix between robot velocities and wheel commands
transform = zeros((8,3))
for i in range(4):
    transform[2*i]=[1, 0, -1*wheel_y_coords[i]]
    transform[2*i+1]=[0, 1, wheel_x_coords[i]]

#Pseudo-inverse for FK
trans_pinv=linalg.pinv(transform)

#Stores the previous direction that each wheel was pointing in
prev_angles = [0, 0, 0, 0]

#Used for floating point comparison
epsilon = .000001

#Estimates linear and angular velocity of robot from wheel angles and speeds
def forward_kinematics(wheel_angles, wheel_speeds):
    
    #Wheel angles and speeds are 4 element lists (see above)
    
    #Create wheel_velocity vector
    wheel_velocities=zeros((8,1))
    for i in range(4):
        wheel_velocities[2*i][0]=wheel_speeds[i]*cos(wheel_angles[i])*(wheel_diameter / 2)
        wheel_velocities[2*i+1][0]=wheel_speeds[i]*sin(wheel_angles[i])*(wheel_diameter / 2)
    
    #Calculate least squares solution
    robot_velocities=dot(trans_pinv, wheel_velocities)
    
    v=sqrt(robot_velocities[0][0]**2+robot_velocities[1][0]**2)
    theta=atan2(robot_velocities[1][0],robot_velocities[0][0])
    omega=robot_velocities[2][0]
    return v, theta, omega

def inverse_kinematics(v, theta, omega):
    """
    Obtain wheel directions and speeds from desired velocities
    
    Inputs:
    v - Linear velocity in m/s
    
    theta - Direction of linear velocity relative to straight ahead in radians
        Negative values of v will result in movement in the opposite direction
    
    omega - Angular velocity in rad/s
    """
    
    #Direction that each wheel needs to point relative to straight ahead in radians
    wheel_angles = [None]*4
    
    #Angular velocity of each wheel in rad/s (Positive is forward)
    wheel_speeds = [None]*4
    
    #Compute wheel velocities in vector form
    velocities = dot(transform,array([[v*cos(theta)],[v*sin(theta)],[omega]]))
    
    for i in range(4):
        vel_x=velocities[2*i][0]
        vel_y=velocities[2*i+1][0]
        wheel_speeds[i]=sqrt(vel_x**2+vel_y**2) / (wheel_diameter / 2)
        
        #Don't bother turning if the wheel isn't moving
        if wheel_speeds[i]<epsilon:
            wheel_angles[i]=prev_angles[i]
            
        #Base case
        elif fabs(vel_x)>epsilon:
            wheel_angles[i]=atan(vel_y/vel_x)
            if vel_x<0:
                wheel_speeds[i]*=-1
                
        #Case where wheel needs to turn 90 degrees from forward
        else:
            wheel_angles[i]=pi/2
            if vel_y<0:
                wheel_speeds[i]*=-1
                
            #turn to other side if it's closer
            if prev_angles[i]<0:
                wheel_speeds[i]*=-1
                wheel_angles[i]*=-1
                
        prev_angles[i] = wheel_angles[i]
        
    return wheel_angles, wheel_speeds