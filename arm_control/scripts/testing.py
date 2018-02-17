#!/usr/bin/env python

# File Name (arm_geometry.py)
# Authors (Patrick Love)

from actuator_controller import JointController
from arm_geometry import *
import math

controller = JointController(
    ActuatorJoint(ARM_ACTUATOR, (3,0), (12,math.radians(5))),
    "encoders/test",
    "drive",
    "control",
    0,
    0.01
    )
controller.Execute("test_control_node")