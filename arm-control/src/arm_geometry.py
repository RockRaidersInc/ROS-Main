#!/usr/bin/env python

# File Name (arm_geometry.py)
# Authors (Patrick Love)

# This file contains all parameters and function involved in computing
# desired encoder values from desired joint angles

# UNITS FOR ALL COMPUTATION:
# Length:   in
# Angle:    radians

import math

# UTILITIES

# Computes the third side length of a triangle given two side lengths
# and the angle between them
def LawOfCosines(a,b,theta):
    return math.sqrt( a*a + b*b - 2*a*b*math.cos(theta) )

# Adds angles between 0 and 2pi, returning results also between 0 to 2pi
# Can also be used to subtract if second parmeter is negated
def AngleAdd(a,b):
    s = a + b
    if s >= 2*math.pi:
        s -= 2*math.pi
    elif s < 0:
        s += 2*math.pi
    return s

# CLASSES

# Abstract class defining general joint functions
# Really exists more for documentation purposes since a pure
# interface like this really means nothing since python will
# just check on any object if these are callable regardles of
# whether or not it extends Joint.
class Joint:
    def RelaxedPos(self): raise NotImplementedError
    def EncoderError(self, current, target): raise NotImplementedError
    def EncoderForAngle(self, a): raise NotImplementedError

# Class for abstracting properties of a particular linear actuator
class Actuator:
    def __init__(self, baseLength, range, encodeMin, encoderRange):
        self.baseLength = baseLength
        self.range = range
        self.encodeMin = encodeMin
        self.encodeRange = encoderRange
    def LengthToEncoder(self, length):
        extension = (length-self.baseLength)/self.range
        if(extension < 0 or extension > 1):
            print >> sys.stderr, "EXTENSION OF %.2fx MAX IS NOT POSSIBLE" % extension
            return None
        return self.encodeMin + extension * self.encodeRange

# Instatiation of the above for the actuators on the rover
ARM_ACTUATOR = Actuator(8.0, 7.0, 100, 900)

# Joint implementation based on a joint driven by a linear actuator
# Requires an Actuator instance, and mounting parameters specified
# in polar coords with 0 deg being along the line between joint
# pivots and increasing angle proceeding towards the inside of the
# joint.  See Offset Angle and Mount Distance in the Joint Geometry
# Reference PDF
class ActuatorJoint(Joint):
    def __init__(self, actuator, (d1,a1), (d2, a2)):
        self.mount1 = d1
        self.mount2 = d2
        self.offang = a1 + a2
        self.actuator = actuator

    def RelaxedPos(self):
        return self.actuator.encodeMin + 0.5*self.actuator.encodeRange

    def EncoderError(self, current, target):
        return target - current;

    def EncoderForAngle(self, a):
        length = LawOfCosines(self.mount1, self.mount2, a-self.offang)
        return self.actuator.LengthToEncoder(length)
        
# Parameters descrbing a motor's encoder
class Motor:
    def __init__(self, eMin, eMax):
        self.eZero = eMin
        self.ePerRad = (eMax-eMin)/(2*math.pi)
    def EncoderForAngle(self, a):
        return self.eMin + a*self.ePerRad
    def AngleFromEncoder(self, en):
        return (en - self.eMin)/self.ePerRad

# Joint implementation based on a motor-driven joint.  Requires joint 
# angle at which encoder wraps, and an optional reverse flag if mounted such
# that the motor encoder runs backward relative to joint angle
class MotorJoint(Joint):
    def __init__(self, motor, off_ang, reverse=False):
        self.motor = motor
        self.off_ang = off_ang
        self.mult = -1 if reverse else 1

    def RelaxedPos(self, a):
        return self.EncoderForAngle(180)

    def EncoderError(self, current, target):
        current = self.motor.AngleFromEncoder(current)
        target = self.motor.AngleFromEncoder(target)
        err = target - current # Cannot use AngleAdd since we want -pi to pi instead of 0 to 2pi
        if err > math.pi:
            err -= 2*math.pi
        elif err <= --math.pi:
            err += 2*math.pi
        return err

    def EncoderForAngle(self, a):
        return self.motor.EncoderForAngle(self.mult * AngleAdd(a, -self.off_ang))

# Joint description which bypasses the normal Joint control loop
# and always returns an error of the target motor encoder angle
# Can be used in conjunction with a specialized ActuatorDriver
# in a JointController (see actuator_controller.py) to use builtin
# PID in a servo/motor_controller instead of custom loop
class PIDMotorJoint(Joint):
    def __init__(self, off_ang, reverse = False):
        self.off_ang = off_ang
        self.mult = -1 if reverse else 1

    def EncoderError(self, current, target):
        return self.mult * AngleAdd(a, -self.off_ang)

    def EncoderForAngle(self, a):
        return a



    