#!/usr/bin/env python

# Author: Robert Katzschmann, Shih-Yuan Liu
# Modified by PCH 2017, David Michelman 2018

gain = 1.0
trim = 0.0
baseline = 1.0  # Distance between wheels (m)
radius = 0.15  # Wheel radius (m)
k = 1.0
limit = 100.0

def forward_kinematics(left, right):
    """
    Convert wheel speeds to car linear and angular velocities.
    """
    # Adjust k by gain and trim
    k_r_inv = (gain + trim) / k
    k_l_inv = (gain - trim) / k

    # Conversion from motor duty to rotation rates
    omega_r = right / k_r_inv
    omega_l = left / k_l_inv

    # Compute linear and angular velocities of car
    v = radius * ( omega_r + omega_l) / 2.0
    omega = radius * (omega_r - omega_l) / baseline

    # Put the car velocities into a message and publish
    return v, omega


def inverse_kinematics(v, omega):
    """
    Convert car linear and angular velocities to wheel speeds.
    """
    # Adjust k by gain and trim
    k_r_inv = (gain + trim) / k
    k_l_inv = (gain - trim) / k

    # Compute motor rotation rates from linear and angular car velocities
    omega_r = (v + 0.5 * omega * baseline) / radius
    omega_l = (v - 0.5 * omega * baseline) / radius

    # Convert from motor rotation rates to duty
    u_r = omega_r * k_r_inv
    u_l = omega_l * k_l_inv

    # Limit output
    u_r_limited = max(min(u_r, limit), -limit)
    u_l_limited = max(min(u_l, limit), -limit)

    # Put the wheel speeds in a message and publish
    return u_l_limited, u_r_limited