#!/usr/bin/env python

import numpy as np
from Jimbo_Kinematics import Inverse_Kin_No_EE as ik
from Jimbo_Kinematics import Forward_Kin_No_EE as fk

angles = ik.inverse_kin(np.array([[0],[32.183],[20.5]]))
print(angles)

print(fk.forward_kin(angles[0][0], angles[1][0], angles[2][0]))
