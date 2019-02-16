from tf.transformations import euler_from_quaternion, quaternion_from_euler
from math import pi


def quat_to_rpy(orientation):
    b = orientation.x
    c = orientation.y
    d = orientation.z
    a = orientation.w

    # roll  = atan2(2*c*a + 2*b*d, 1 - 2*c*c - 2*d*d) * 180 / pi
    # pitch = atan2(2*b*a + 2*c*d, 1 - 2*b*b - 2*d*d) * 180 / pi
    # yaw   = asin(2*b*c + 2*d*a) * 180 / pi

    # R = [[2*(a**2 + b**2) - 1, 2*(b*c - a*d),       2*(b*d + a*c)],
    #      [2*(b*c + a*d),       2*(a**2 + c**2) - 1, 2*(c*d - a*b)],
    #      [2*(b*d - a*c),       2*(c*d + a*b),       2*(a**2 + d**2) - 1]]

    # sy = (R[0][0] * R[0][0] + R[1][0] * R[1][0]) ** (1./2)

    # roll = atan2(R[2][1], R[2][2])
    # pitch = atan2(-R[2][0], sy)
    # yaw = atan2(R[1][0], R[0][0])

    roll, pitch, yaw = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])

    return roll *180/pi, pitch*180/pi, yaw*180/pi