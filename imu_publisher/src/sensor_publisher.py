#!/usr/bin/env python
import serial
import rospy
import re as reeeeee  # named in honor of RPI sudents
import pdb
import time

from sensor_msgs.msg import Imu as Imu_msg
from sensor_msgs.msg import MagneticField as MagneticField_msg


def main(imu_pub, mag_pub):
    ser = serial.Serial('/dev/serial/by-id/usb-Arduino_Srl_Arduino_Mega_7563331313335180E041-if00', 115200, timeout=1)

    # this is an easy way to parse the data coming out of the IMU. The Arduino prints the data
    # as text instad of in binary so that it would be human readable (and much easier to debug)
    regex = reeeeee.compile(r'Start: (?P<accel_x>[0-9.-]*) (?P<accel_y>[0-9.-]*) (?P<accel_z>[0-9.-]*), (?P<mag_x>[0-9.-]*) (?P<mag_y>[0-9.-]*) (?P<mag_z>[0-9.-]*), (?P<gyro_x>[0-9.-]*) (?P<gyro_y>[0-9.-]*) (?P<gyro_z>[0-9.-]*)[ ]*[\\r]*[\\n]*')

    while True:
        # pdb.set_trace()  # for debugging
        try:
            line = ser.readline()
        except serial.serialutil.SerialException as e:
            print("got a SerialException: ", e)
            
        match = regex.search(line)
        print(line)
        print(match)
        if match is None:
            continue  # The arduino prints out some non-data text when it starts up, 
                        # we probably read that. nothing to do except read more data   

        try:
            imu_msg = Imu_msg()
            mag_msg = MagneticField_msg()
            imu_msg.linear_acceleration.x = float(match.group('accel_x'))
            imu_msg.linear_acceleration.y = float(match.group('accel_y'))
            imu_msg.linear_acceleration.z = float(match.group('accel_z'))
            imu_msg.angular_velocity.x = float(match.group('gyro_x'))
            imu_msg.angular_velocity.y = float(match.group('gyro_y'))
            imu_msg.angular_velocity.z = float(match.group('gyro_z'))

            mag_msg.magnetic_field.x = float(match.group('mag_x'))
            mag_msg.magnetic_field.y = float(match.group('mag_y'))
            mag_msg.magnetic_field.z = float(match.group('mag_z'))

            imu_pub.publish(imu_msg)
            mag_pub.publish(mag_msg)
        except ValueError as e:
            pass  # there was probably some corrupted data from the serial connection 
        
        # time.sleep(0.1)  # who needs rate limiting?


if __name__ == "__main__":
    imu_pub = rospy.Publisher('imu/data_raw', Imu_msg, queue_size = 1)
    mag_pub = rospy.Publisher('imu/mag', MagneticField_msg, queue_size = 1)
    rospy.init_node('IMU_reader_node')
    main(imu_pub, mag_pub)
