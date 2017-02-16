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
t = numpy.zeros((1,3))
t[0,0] = 0.2
t[0,1] = 0.3
t[0,2] = -0.9
xy_ideal = numpy.matrix([[-1.0001], [-1.0001]])
damping = 0.010

# robot arm joint lengths
L = numpy.zeros(3, dtype=float)
L[0] = 1.0
L[1] = 0.75
L[2] = 0.5

v = 0.25  # target end effector velocity
fps = 30
screen = None



def Setup():
    pygame.init()
    screen = pygame.display.set_mode((600, 600))



def Loop():
    global t
    dx = v / fps

    xy_curr = numpy.matrix([[L[0] * math.cos(t[0]) + L[1] * math.cos(t[0]+t[1])], [L[0] * math.sin(t[0]) + L[1] * math.sin(t[0]+t[1])]])
    # calculate a jacobian matrix
    J = numpy.matrix([[-1*L[0]*math.sin(t[0]) + -1*L[1]*math.sin(t[0]+t[1]), -1*L[1]*math.sin(t[0]+t[1])],
                      [L[0]*math.cos(t[0]) + L[1]*math.cos(t[0]+t[1]), L[1]*math.cos(t[0]+t[1])]])

    xy_curr2 = calculate_forward_kinematics(t)
    J2 = calculate_jacobian(calculate_forward_kinematics, t)
    print "theta: ", t, "\nxy_curr: ", xy_curr
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
        dt = dt / numpy.linalg.norm(dt) * dx
    else:
        dt = dt * numpy.linalg.norm(dt)  # exponentially approach target position
    t = t + dt

    # xy_j1 = numpy.matrix([[L[0] * math.cos(t[0])], [-1*L[0] * math.sin(t[0])]])
    # xy_curr[1] *= -1

    x_curr, y_curr = 300, 300
    current_reference_angle = 0
    scale = 100
    pygame.draw.rect(screen, (255, 255, 255), (0, 0, 600, 600))
    pygame.draw.circle(screen, (0, 128, 255), (x_curr, y_curr), 10)
    for i in range(L.shape[0]):
        current_reference_angle += t[i, 0]
        x_new = x_curr + L[i] * math.cos(current_reference_angle) * scale
        y_new = y_curr + L[i] * math.sin(current_reference_angle) * scale * -1
        pygame.draw.line(screen, (0, 128, 255), (int(x_curr), int(y_curr)), (int(x_new), int(y_new)), 5)
        pygame.draw.circle(screen, (0, 128, 255), (int(x_new), int(y_new)), 10)
        x_curr = x_new
        y_curr = y_new

    pygame.draw.circle(screen, (0, 255, 128), (int(xy_ideal[0] * scale + 300), int(xy_ideal[1] * scale * -1 + 300)), 20)

    pygame.display.flip()



def calculate_forward_kinematics(curr_angles):
    assert(L.shape[0] == curr_angles.shape[0])
    end_position = numpy.zeros(2, dtype=float)
    current_reference_angle = 0
    for i in range(L.shape[0]):
        current_reference_angle += curr_angles[i, 0]
        end_position[0] += L[i] * math.cos(current_reference_angle)
        end_position[1] += L[i] * math.sin(current_reference_angle)
    return numpy.matrix(end_position).transpose()


# calcualte the jacobian of a functhion, f, at point current_input
def calculate_jacobian(f, current_input, dx = 0.01):
    output = numpy.zeros((f(current_input).transpose().shape[1], current_input.shape[0]), dtype=float)
    delta = numpy.zeros(current_input.shape)
    for i in range(current_input.shape[0]):
        delta[i] += dx
        differences = (f(current_input + delta).transpose() - f(current_input - delta).transpose()) / (dx * 2)
        delta[i] -= dx
        for j in range(output.shape[0]):
            output[j, i] = differences[0, j]
    return numpy.matrix(output)


if __name__ == "__main__":
    Setup()
    while(True):
        last_time = time.time()
        Loop()
        while (time.time() - last_time < 1 / fps):
            time.sleep(0.001)
