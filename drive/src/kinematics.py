#!/usr/bin/env python

#Last edited by Connor McGowan 1/18/19

#These should go in a config file
track = 35.485/12  # Horizontal distance between wheels (ft)
diameter = 1.0  # Wheel diameter (ft)

def forward_kinematics(left, right):
    """
    Convert wheel speeds to vehicle linear and angular velocities.
    """

    # Compute linear and angular velocities of car
    v = (left+right)*diameter/4
    omega = (right - left)*diameter/(2*track)

    # Put the car velocities into a message and publish
    return v, omega


def inverse_kinematics(v, omega):
    """
    Convert vehicle linear and angular velocities to wheel speeds.
    """

    # Compute motor rotation rates from linear and angular car velocities
    omega_r = (2*v+track*omega)/diameter
    omega_l = (2*v-track*omega)/diameter

    # Put the wheel speeds in a message and publish
    return omega_l, omega_r
