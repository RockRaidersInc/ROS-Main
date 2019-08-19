#!/usr/bin/env python

#Last edited by Connor McGowan 8/19/19

from math import pi, sqrt, sin, cos, atan, fabs

#These should go in a config file
#These values define the dimensions of the rover, as well as the location of the
#point where the velocities are measured (reference point)

#Diameter of wheel, in meters
wheel_diameter = 2

#Distance between wheels on opposite sides of the chassis, in meters
track = 1

#Distance from the reference point to the line passing through both front wheels, in meters
#If the reference point is in front of the front wheels, this number will be negative
front_wheelbase = 2

#Distance from the reference point to the line passing through both rear wheels, in meters
#If the reference point is behind the rear wheels, this number will be negative
rear_wheelbase = 3

#Stores the previous direction that each wheel was pointing in
prev_angles = [0, 0, 0, 0]

#Used for floating point comparison
epsilon = .000001

def inverse_kinematics(v, theta, omega):
    """
    Obtain wheel directions and speeds from desired velocities
    
    Inputs:
    v - Linear velocity in m/s
    
    theta - Direction of linear velocity relative to straight ahead in radians
        Negative values of v will result in movement in the opposite direction
    
    omega - Angular velocity in rad/s
    """
    
    #Constrain angle to (-pi/2, pi/2]
    while theta <= -pi/2:
        theta += pi
        v *= -1
    while theta > pi/2:
        theta -= pi
        v *= -1
    
    #Establish coordinates of the center of each wheel relative to the reference point
    wheel_x_coords = [track/-2.0, track/2.0, track/-2.0, track/2.0]
    wheel_y_coords = [front_wheelbase, front_wheelbase, -1*rear_wheelbase, -1*rear_wheelbase]
    
    #Distances from each wheel to the center of rotation (m)
    wheel_turn_radii = [None]*4
    
    #Direction that each wheel needs to point relative to straight ahead in radians
    wheel_angles = [None]*4
    
    #Angular velocity of each wheel in rad/s (Positive is forward)
    wheel_speeds = [None]*4
    
    #Case where center of rotation is at infinity
    if omega==0:
        for i in range(4):
            wheel_angles[i] = theta
            wheel_speeds[i] = v / (wheel_diameter / 2)
    
    else:
    
        #Find coordinates of the center of rotation
        turn_radius = v / omega
        center_x = -1 * turn_radius * cos(theta)
        center_y = -1 * turn_radius * sin(theta)
        
        for i in range(4):
            wheel_turn_radii[i] = sqrt((center_y - wheel_y_coords[i])**2 + (center_x - wheel_x_coords[i])**2)
            wheel_speeds[i] = omega * wheel_turn_radii[i] / (wheel_diameter / 2)
            
            #Reverse direction when the center of rotation is on right side of wheel
            if center_x - wheel_x_coords[i] > epsilon:
                wheel_speeds[i] *= -1
                
            #Case where wheel must turn 90 degrees
            if fabs(wheel_x_coords[i] - center_x) <= epsilon:
                wheel_angles[i] = pi/2
                if center_y > wheel_y_coords[i]:
                    wheel_speeds[i] *= -1
                    
            else:
                wheel_angles[i] = atan((center_y - wheel_y_coords[i]) / (center_x - wheel_x_coords[i]))
    
    for i in range(4):
    
        #Don't bother turning the wheel if it's not moving
        if fabs(wheel_speeds[i]) <= epsilon:
            wheel_angles[i] = prev_angles[i]
            
        #For cases where wheel is at 90 degrees, go to the other side if it's closer
        elif wheel_angles[i] == pi/2 and prev_angles[i] < 0:
            wheel_angles[i] *= -1
            wheel_speeds[i] *= -1
            
        prev_angles[i] = wheel_angles[i]
        
    return wheel_angles, wheel_speeds