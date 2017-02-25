##############################################################################

# inverse_kinematics.py
# Authors: David Michelman (miched@rpi.edu)


# Description (what does this node do?)
#TODO:

# This code shouldn't be called directly, more documentation to come

# Dependencies:
# This code depends on Numpy, and the graphics portion depends on pygame.

##############################################################################

import numpy
import math
import pygame
import time


#TODO: cleanup
#CONSTANTS
#names for constants should be in ALL CAPS
# robot arm starting joint angles
t = None
# t = numpy.zeros((1,3))
# t[0,0] = 0.2
# t[0,1] = 0.3
# t[0,2] = -0.9
# xy_ideal = numpy.matrix([[-1.0001], [-1.0001]])
damping = 0.010

# robot arm joint lengths
ARM_LENGTHS = numpy.zeros(3, dtype=float)
ARM_LENGTHS[0] = 1.0
ARM_LENGTHS[1] = 0.75
ARM_LENGTHS[2] = 0.5

time = time.time()


v = 0.25  # target end effector velocity
fps = 30
screen = None


# sets the joint lengths of the arm.
def Setup(joint_lengths_in):
    global ARM_LENGTHS
    ARM_LENGTHS = joint_lengths_in


# Calculate new joint angles for the precision manipulator. Takes the current joint angles (in order from the robot
# to the end effector) and the target position (x, y, z). It returns new joint in the same format as they were passed
# in in.
def Loop(current_angles, target_position):
    global time
    last_time = time
    time = time.time()
    dt = last_time - time
    dx = v * dt

    t = numpy.matrix(current_angles).transpose()
    xy_ideal = numpy.matrix(target_position).transpose()

    # xy_curr = numpy.matrix([[L[0] * math.cos(t[0]) + L[1] * math.cos(t[0]+t[1])], [L[0] * math.sin(t[0]) + L[1] * math.sin(t[0]+t[1])]])
    # # calculate a jacobian matrix
    # J = numpy.matrix([[-1*L[0]*math.sin(t[0]) + -1*L[1]*math.sin(t[0]+t[1]), -1*L[1]*math.sin(t[0]+t[1])],
    #                   [L[0]*math.cos(t[0]) + L[1]*math.cos(t[0]+t[1]), L[1]*math.cos(t[0]+t[1])]])

    xy_curr2 = calculate_forward_kinematics(t)
    J2 = calculate_jacobian(calculate_forward_kinematics, t, ARM_LENGTHS)
    # print "theta: ", t, "\nxy_curr: ", xy_curr
    xy_curr = xy_curr2
    J = J2

    # pseudoinverse method
    # J_inv = numpy.linalg.pinv(J)
    # # J_inv = numpy.transpose(J)
    # J_inv = numpy.clip(J_inv, -10, 10)
    # dt = J_inv * (xy_ideal - xy_curr) * dx

    # damped transpose method
    j_t = numpy.transpose(J)
    # j_t2 = numpy.transpose(J2)
    dt = j_t * numpy.linalg.inv(J * j_t + damping**2 * numpy.identity(2)) * (xy_ideal - xy_curr)

    if numpy.linalg.norm(dt) > 0.01:
        dt /= numpy.linalg.norm(dt) * dx
    else:
        dt *= numpy.linalg.norm(dt)  # exponentially approach target position
    t += dt

    # xy_j1 = numpy.matrix([[L[0] * math.cos(t[0])], [-1*L[0] * math.sin(t[0])]])
    # xy_curr[1] *= -1
    return list(t.transpose())


# Calculates the 3 dimensional position of the end effector of the arm from passed joint angles and lengths.
#TODO: test in 3 dimensions
def calculate_forward_kinematics(curr_angles, Lengths):
    end_position = numpy.zeros(3, dtype=float)
    current_elevation_angle = 0
    current_yaw_angle = curr_angles[0]
    for i in range(1, Lengths.shape[0]):
        current_elevation_angle += curr_angles[i, 0]
        unit_v = numpy.matrix([math.sin(current_yaw_angle), math.cos(current_yaw_angle), 0])
        unit_v[0] *= math.cos(current_elevation_angle)
        unit_v[1] *= math.cos(current_elevation_angle)
        unit_v[3] = math.sin(current_elevation_angle)
        end_position += unit_v * Lengths[i]

    return numpy.matrix(end_position).transpose()


# calculate the jacobian of a functhion, f, at point current_input (and pass joint_lengths into f as a second argument)
def calculate_jacobian(f, current_input, joint_lengths, dx = 0.01):
    output = numpy.zeros((f(current_input).transpose().shape[1], current_input.shape[0]), dtype=float)
    delta = numpy.zeros(current_input.shape)
    for i in range(current_input.shape[0]):
        delta[i] += dx
        differences = (f(current_input + delta, joint_lengths).transpose() -
                       f(current_input - delta, joint_lengths).transpose()) / (dx * 2)
        delta[i] -= dx
        for j in range(output.shape[0]):
            output[j, i] = differences[0, j]
    return numpy.matrix(output)


def draw_graphics():
    x_curr, y_curr = 300, 300
    current_reference_angle = 0
    scale = 100
    pygame.draw.rect(screen, (255, 255, 255), (0, 0, 600, 600))
    pygame.draw.circle(screen, (0, 128, 255), (x_curr, y_curr), 10)
    for i in range(ARM_LENGTHS.shape[0]):
        current_reference_angle += t[i, 0]
        x_new = x_curr + ARM_LENGTHS[i] * math.cos(current_reference_angle) * scale
        y_new = y_curr + ARM_LENGTHS[i] * math.sin(current_reference_angle) * scale * -1
        pygame.draw.line(screen, (0, 128, 255), (int(x_curr), int(y_curr)), (int(x_new), int(y_new)), 5)
        pygame.draw.circle(screen, (0, 128, 255), (int(x_new), int(y_new)), 10)
        x_curr = x_new
        y_curr = y_new

    # pygame.draw.circle(screen, (0, 255, 128), (int(xy_ideal[0] * scale + 300), int(xy_ideal[1] * scale * -1 + 300)), 20)

    pygame.display.flip()


if __name__ == "__main__":
    Setup()
    pygame.init()
    screen = pygame.display.set_mode((600, 600))
    while(True):
        last_time = time.time()
        Loop()
        draw_graphics()
        while (time.time() - last_time < 1 / fps):
            time.sleep(0.001)
