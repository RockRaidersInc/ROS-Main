#Contains helper functions to convert between angles and linear actuator
#distances for the second and third joint of the arm
#Joint limits are set in the convert_to_distance function

import numpy as np
import Jimbo_Kinematics.Kinematic_Utils as ku


#converts distances from zero position in inches to angle from zero position in radians
#input: 2x1 vector (numpy array) of actuator offsets (distance from zero position)
#[[d1]
# [d2]]
#output: 2x1 vector (numpy array) of joint angle offsets

def convert_to_angle(x):
    #Just some trig stuff
    q1 = np.pi/2-ku.q2_zero-np.arctan(1.0/40.0)-np.arctan(14.0/19.0)-np.arccos((435.0625-(x[0][0]+ku.LA2_ZERO_POSITION)**2)/(2*np.sqrt(13933.70313)))
    q2 = np.arctan(1.0/37.0) + np.arccos((388.0625-(x[1][0]+ku.LA3_ZERO_POSITION)**2)/(13.5*np.sqrt(342.5)))-np.pi/2.0-ku.q3_zero
    return np.array([[q1],[q2]])


#converts angle from zero position in radians to distances from zero position in inches
#input: 2x1 vector (numpy array) of actuator angles (measured from zero position)
#[[q1]
# [q2]]
#output: 2x1 vector (numpy array) of linear actuator distances
#Will return Nan if joint is past limits
    
def convert_to_distance(q):
    #Literally just the inverse of the other function (so more trig)
    #actuator 1 angle limits
    if q[0][0]+ku.q2_zero<.2912 and q[0][0]+ku.q2_zero>-2.3213: 
        x1 = np.sqrt(435.0625-np.cos(np.pi/2-ku.q2_zero-np.arctan(1.0/40.0)-np.arctan(14.0/19.0)-q[0][0])*2*np.sqrt(13933.70313))-ku.LA2_ZERO_POSITION
    else:
        x1 = float('nan')
    #actuator 2 angle limits
    if q[1][0]+ku.q3_zero>-.5359 and q[1][0]+ku.q3_zero<1.085: 
        x2 = np.sqrt(388.0625-np.cos(np.pi/2+ku.q3_zero-np.arctan(1.0/37.0)+q[1][0])*13.5*np.sqrt(342.5))-ku.LA3_ZERO_POSITION
    else:
        x2 = float('nan')
    return np.array([[x1],[x2]])

#Move to config file?
SHOULDER_MIN_EXTENSION = 0
SHOULDER_MAX_EXTENSION = 10
SHOULDER_MIN_READING = 0
SHOULDER_MAX_READING = 1024

ELBOW_MIN_EXTENSION = 0
ELBOW_MAX_EXTENSION = 10
ELBOW_MIN_READING = 0
ELBOW_MAX_READING = 1024

def map(val, in_min, in_max, out_min, out_max):
    return (val - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def shoulder_dist_to_enc(s):
    return map(s, SHOULDER_MIN_EXTENSION, SHOULDER_MAX_EXTENSION, SHOULDER_MIN_READING, SHOULDER_MAX_READING)
def elbow_dist_to_enc(e):
    return map(e, ELBOW_MIN_EXTENSION, ELBOW_MAX_EXTENSION, ELBOW_MIN_READING, ELBOW_MAX_READING)
def shoulder_enc_to_dist(s):
    return map(s, SHOULDER_MIN_READING, SHOULDER_MAX_READING, SHOULDER_MIN_EXTENSION, SHOULDER_MAX_EXTENSION)
def elbow_enc_to_dist(e):
    return map(e, ELBOW_MIN_READING, ELBOW_MAX_READING, ELBOW_MIN_EXTENSION, ELBOW_MAX_EXTENSION)

