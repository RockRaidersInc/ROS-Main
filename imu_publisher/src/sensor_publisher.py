#!/usr/bin/env python
import serial
import rospy
import re as reeeeee  # named in honor of RPI sudents
import pdb
import time
import sys
import signal

from sensor_msgs.msg import Imu as Imu_msg
from sensor_msgs.msg import MagneticField as MagneticField_msg


arduino_serial_id = "usb-Arduino__www.arduino.cc__0043_75335313437351E01021-if00"


def signal_handler(sig, frame):
    print('control c detected, exiting')
    sys.exit(0)

def set_covariance_as_identity(val):
    for i in range(9):
        val[i] = 0
    val[0] = 1
    val[4] = 1
    val[8] = 1


def main(imu_pub, mag_pub):

    signal.signal(signal.SIGINT, signal_handler)

    expected_seq_num = None
    arduino_start_time = time.time()

    if len(sys.argv) > 1 and sys.argv[1] == "no_print":
        no_print = True
    else:
        no_print = False

    ser = serial.Serial('/dev/serial/by-id/' + arduino_serial_id, 115200, timeout=1)

    # this is an easy way to parse the data coming out of the IMU. The Arduino prints the data
    # as text instad of in binary so that it would be human readable (and much easier to debug)
    # regex = reeeeee.compile(r'Start:[ ]+(?P<accel_x>[0-9.-]*)[ ]+(?P<accel_y>[0-9.-]*)[ ]+(?P<accel_z>[0-9.-]*),[ ]+(?P<mag_x>[0-9.-]*)[ ]+(?P<mag_y>[0-9.-]*)[ ]+(?P<mag_z>[0-9.-]*),[ ]+(?P<gyro_x>[0-9.-]*)[ ]+(?P<gyro_y>[0-9.-]*)[ ]+(?P<gyro_z>[0-9.-]*)[ ]*[\\r]*[\\n]*')
    regex = reeeeee.compile(r'Seq:(?P<sequence_num>[0-9]+), ms_time:(?P<timestamp>[0-9]+),[ ]+(?P<accel_x>[0-9.-]*)[ ]+(?P<accel_y>[0-9.-]*)[ ]+(?P<accel_z>[0-9.-]*),[ ]+(?P<mag_x>[0-9.-]*)[ ]+(?P<mag_y>[0-9.-]*)[ ]+(?P<mag_z>[0-9.-]*),[ ]+(?P<gyro_x>[0-9.-]*)[ ]+(?P<gyro_y>[0-9.-]*)[ ]+(?P<gyro_z>[0-9.-]*)[ ]*[\\r]*[\\n]*')
    while True:
        # pdb.set_trace()  # for debugging
        try:
            line = ser.readline()
        except serial.serialutil.SerialException as e:
            print("got a SerialException: ", e)
            continue
            
        match = regex.search(line)
        if not no_print:
            print(line)
            print(match)
        if match is None:
            continue  # The arduino prints out some non-data text when it starts up, 
                        # we probably read that. nothing to do except read more data   
        if not no_print:
            print(match.group('gyro_x'))

        recieved_seq_num = float(match.group('sequence_num'))
        recieved_timestamp = float(match.group('timestamp')) / 1000.  # convert from ms to seconds

        if expected_seq_num is None or recieved_seq_num < expected_seq_num:
            # the arduino was power-cycled since data was last recieved, reset the time offset. 
            arduino_start_time = time.time() - recieved_timestamp
            
        expected_seq_num = recieved_seq_num + 1
        data_collection_time = arduino_start_time + recieved_timestamp

        try:
            #TODO: add a timestamp 
            imu_msg = Imu_msg()
            imu_msg.header.stamp = rospy.get_rostime()
            imu_msg.header.stamp.secs = int(data_collection_time)
            imu_msg.header.stamp.nsecs = int((data_collection_time - imu_msg.header.stamp.secs) * 1000000000)
            
            imu_msg.linear_acceleration.x = float(match.group('accel_x'))
            imu_msg.linear_acceleration.y = float(match.group('accel_y'))
            imu_msg.linear_acceleration.z = float(match.group('accel_z'))
            set_covariance_as_identity(imu_msg.linear_acceleration_covariance)
            imu_msg.angular_velocity.x = float(match.group('gyro_x'))
            imu_msg.angular_velocity.y = float(match.group('gyro_y'))
            imu_msg.angular_velocity.z = float(match.group('gyro_z'))
            set_covariance_as_identity(imu_msg.angular_velocity_covariance)

            mag_msg = MagneticField_msg()
            mag_msg.header.stamp = rospy.get_rostime()
            mag_msg.header.stamp.secs = int(data_collection_time)
            mag_msg.header.stamp.nsecs = int((data_collection_time - imu_msg.header.stamp.secs) * 1000000000)
            mag_msg.magnetic_field.x = float(match.group('mag_x'))
            mag_msg.magnetic_field.y = float(match.group('mag_y'))
            mag_msg.magnetic_field.z = float(match.group('mag_z'))
            set_covariance_as_identity(mag_msg.magnetic_field_covariance)

            imu_pub.publish(imu_msg)
            mag_pub.publish(mag_msg)
        except ValueError as e:
            pass  # there was probably some corrupted data from the serial connection 
        
        # time.sleep(0.1)  # who needs rate limiting?


if __name__ == "__main__":
    imu_pub = rospy.Publisher('imu/data_raw', Imu_msg, queue_size = 2)
    mag_pub = rospy.Publisher('imu/mag', MagneticField_msg, queue_size = 2)
    rospy.init_node('IMU_reader_node')
    main(imu_pub, mag_pub)
