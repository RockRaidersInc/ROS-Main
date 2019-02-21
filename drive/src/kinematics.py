#!/usr/bin/env python

#Last edited by Connor McGowan 1/18/19  -- no longer true, see David

from math import sin, atan2

#These should go in a config file
track_default = 0.901  # Horizontal distance between wheels (ft)
diameter_default = 0.305  # Wheel diameter (ft)
track_length_default = 0.762  # distance between front and back wheels

def forward_kinematics(left, right, track=track_default, diameter=diameter_default, track_length=track_length_default):
    """
    Convert wheel speeds to vehicle linear and angular velocities.
    """
    # Compute linear and angular velocities of car
    v = (left + right) * diameter / 4
    omega = (right - left) * diameter / (2 * track)
    theta = atan2(track, track_length)
    omega_component = omega * sin(theta)

    # Put the car velocities into a message and publish
    return v, omega_component


def inverse_kinematics(v, omega_raw, track=track_default, diameter=diameter_default, track_length=track_length_default):
    """
    Convert vehicle linear and angular velocities to wheel speeds.
    """
    theta = atan2(track, track_length)
    omega = omega_raw / sin(theta)
    # Compute motor rotation rates from linear and angular car velocities
    omega_l = (2*v-track*omega)/diameter
    omega_r = (2*v+track*omega)/diameter

    # Put the wheel speeds in a message and publish
    return omega_l, omega_r
