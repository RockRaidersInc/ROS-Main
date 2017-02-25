#!/usr/bin/env python

# This file contains all parameters and function involved in computing
# the linear actuator positions from desired arm angles

# UNITS FOR ALL COMPUTATION:
# Length:   in
# Angle:    radians

import math

# CLASSES

class MountingParams
    def __init__():
        self.mount1 = None
        self.mount2 = None
        self.offang1 = 0
        self.offang2 = 0
    def firstMount(dist, ang):
        self.mount1 = dist
        self.offang1 = ang
        return self
    def secondMount(dist, ang):
        self.mount1 = dist
        self.offang1 = ang
        return self
    def totalOffset():
        return self.offang1+self.offang2

class ActuatorParams
    def __init__(baseLength, range, encodeMin, encoderRange):
        self.baseLength = baseLength
        self.range = range
        self.encodeMin = encodeMin
        self.encodeRange = encoderRange

#*********** ARM PARAMETERS ***********
# Currently estimated, should be filled in more precisely when arm design
# is finalized
SHOULDER_MOUNT = MountingParams() \
    .firstMount(    3.0,     math.radians(45)) \
    .secondMount(   12.0,    math.radians(5.0))

ELBOW_MOUNT = MountingParams() \
    .firstMount(    12.0,    math.radians(-3.0)) \
    .secondMount(   -3.0,    0)

WRIST_MOUNT = MountingParams() \
    .firstMount(    12.0,    math.radians(3.0)) \
    .secondMount(    2.0,    math.radians(90.0))

#********* ACTUATOR PARAMETERS *********
#Unsure as yet if actuator manufacturing is consistent enough that we can
#just provide one set, or if we should measure each individually.  I expect
#base length and range are consistent, encoder values I am not as sure.
#Only one set provided for now
ACTUATOR = ActuatorParams(8.0, 7.0, 0.0, 12.0)


# Computes the third side length of a triangle given two side lengths
# and the angle between them
def LawOfCosines(a,b,theta):
    return math.sqrt( a*a + b*b - 2*a*b*math.cos(theta) )

# Uses mounting parameters to comupute the total linear actuator length
# required to get the supplied arm angle
# angle -        Desired joint angle
# mount1 -        Distance from joint rotation axis to first actuator mounting pin
# mount2 -        Distance from joint rotation axis to second actuator mounting pin
# offsetAngle -    Sum of the angles 
def LengthFromAngle(angle, mountParams):
    return LawOfCosines(mountParams.mount1, mountParams.mount2, angle-mountParams.totalOffset())

# Computes the encoder value associated with the actuator spanning a specifed
# total length.  If the actuator cannot physically produce the given length,
# None is returned and an error is printed to stderr
def ActuatorPositionForLength(totalLength, actuatorParams ):
    extensionFactor = (totalLength-actuatorParams.baseLength)/actuatorParams.range
    if(extensionFactor < 0 or extensionFactor > 1):
        print >> sys.stderr, "EXTENSION OF %.2fx MAX IS NOT POSSIBLE" % extensionFactor
        return None
    return actuatorParams.encodeMin + extensionFactor * actuatorParams.encodeRange


def PositionForAngle(angle, mount, actuator):
    actuatorLength = LengthFromAngle(angle, mount)
    actuatorExtension = ActuatorPositionForLength(actuatorLength, actuator)
    return actuatorExtension



    