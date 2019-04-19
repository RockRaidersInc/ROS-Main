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
from nmea_msgs.msg import Sentence as NmeaSentence_msg


arduino_serial_id = "usb-Arduino__www.arduino.cc__0043_75335313437351E01021-if00"


def signal_handler(sig, frame):
    print('control c detected, exiting')
    sys.exit(0)

def set_covariance(val, cov):
    for i in range(9):
        val[i] = 0
    val[0] = cov
    val[4] = cov
    val[8] = cov


def get_line_from_list(chars):
    try:
        end_position = chars.index("\n")
        return chars[end_position+1:], "".join(chars[0:end_position+1])
    except ValueError:
        return chars, None


arduino_start_time  = 0.0
no_print = False

def main(imu_pub, mag_pub):
    global arduino_start_time
    global no_print

    signal.signal(signal.SIGINT, signal_handler)


    arduino_start_time = time.time()

    if len(sys.argv) > 1 and sys.argv[1] == "no_print":
        no_print = True
    else:
        no_print = False

    ser = serial.Serial('/dev/serial/by-id/' + arduino_serial_id, 19200, timeout=1)
    # ser = serial.Serial('/dev/serial/by-id/' + arduino_serial_id, 115200, timeout=1)

    # this is an easy way to parse the data coming out of the IMU. The Arduino prints the data
    # as text instad of in binary so that it would be human readable (and much easier to debug)
    regex = reeeeee.compile(r'Seq:(?P<sequence_num>[0-9]+), ms_time:(?P<timestamp>[0-9]+),[ ]+(?P<accel_x>[0-9.-]*)[ ]+(?P<accel_y>[0-9.-]*)[ ]+(?P<accel_z>[0-9.-]*),[ ]+(?P<mag_x>[0-9.-]*)[ ]+(?P<mag_y>[0-9.-]*)[ ]+(?P<mag_z>[0-9.-]*),[ ]+(?P<gyro_x>[0-9.-]*)[ ]+(?P<gyro_y>[0-9.-]*)[ ]+(?P<gyro_z>[0-9.-]*)[ ]*[\\r]*[\\n]*')

    # there are two data streams coming off of the arduino simultaneously. The IMU produces more data so it sends
    # characters normally. The GPS doesn't send as many characters, so any data from it is prefixed with a !
    imu_buffer = list()
    gps_buffer = list()
    input_buffer = list()   # for non-sorted data

    while True:
        # pdb.set_trace()  # for debugging

        try:
            bytes_to_read = ser.inWaiting()
            if bytes_to_read > 0:
                input_bytes = ser.read(bytes_to_read)
                for i in input_bytes:
                    input_buffer.append(i)
            else:
                time.sleep(0.005)  # sleep a few milliseconds, there is no data
        except serial.serialutil.SerialException as e:
            print("got a SerialException: ", e)
            continue

        while len(input_buffer) > 0:
            if input_buffer[0] != "!":
                imu_buffer.append(input_buffer.pop(0))
            elif input_buffer[0] == "!" and len(input_buffer) == 1:
                # if the only character left in input_buffer is "!" then the next character belongs in the gpu buffer,
                # but it hasn't been read yet. Stop sorting data and wait for the next character
                break
            else:
                input_buffer.pop(0)
                gps_buffer.append(input_buffer.pop(0))

        imu_buffer, imu_line = get_line_from_list(imu_buffer)
        if imu_line is not None:
            # print(imu_line)
            process_imu_data(imu_line, regex)

        gps_buffer, gps_line = get_line_from_list(gps_buffer)
        if gps_line is not None:
            # print("gps_line:", gps_line)
            send_gps_nmea_sentence(gps_line)


def send_gps_nmea_sentence(sentence):
    sentence = sentence.strip()
    msg = NmeaSentence_msg()
    msg.sentence = sentence
    msg.header.stamp = rospy.get_rostime()
    msg.header.frame_id = 'gps0_link'
    gps_pub.publish(msg)


expected_seq_num = None

def process_imu_data(line, regex):
    global expected_seq_num
    global arduino_start_time
    global no_print

    match = regex.search(line)
    # if not no_print:
    #     print(line)
    #     print(match)
    if match is None:
        return  # The arduino prints out some non-data text when it starts up,
        # we probably read that. nothing to do except read more data

    recieved_seq_num = float(match.group('sequence_num'))
    recieved_timestamp = float(match.group('timestamp')) / 1000.  # convert from ms to seconds

    if expected_seq_num is None or recieved_seq_num < expected_seq_num:
        # the arduino was power-cycled since data was last recieved, reset the time offset.
        arduino_start_time = time.time() - recieved_timestamp

    expected_seq_num = recieved_seq_num + 1
    data_collection_time = arduino_start_time + recieved_timestamp

    try:
        # TODO: add a timestamp  -- edit, I think it's done? it would probably be good to check
        imu_msg = Imu_msg()
        imu_msg.header.stamp = rospy.get_rostime()
        imu_msg.header.stamp.secs = int(data_collection_time)
        imu_msg.header.stamp.nsecs = int((data_collection_time - imu_msg.header.stamp.secs) * 1000000000)
        imu_msg.header.frame_id = "imu0_link"

        accel_x, accel_y, accel_z = float(match.group('accel_x')), float(match.group('accel_y')), float(match.group('accel_z'))
        imu_msg.linear_acceleration.x = map_to(accel_x, -0.4, 10.3, 0, 9.81)
        imu_msg.linear_acceleration.y = map_to(accel_y, 0.2, 9.7, 0, 9.81)
        imu_msg.linear_acceleration.z = map_to(accel_z, 0, 10.75, 0, 9.81)
        set_covariance(imu_msg.linear_acceleration_covariance, 1.0)

        gyro_x, gyro_y, gyro_z = float(match.group('gyro_x')), float(match.group('gyro_y')), float(match.group('gyro_z'))
        imu_msg.angular_velocity.x = gyro_x - 0.01
        imu_msg.angular_velocity.y = gyro_y + 0.01
        imu_msg.angular_velocity.z = gyro_z - 0.04
        set_covariance(imu_msg.angular_velocity_covariance, 1.0)

        mag_msg = MagneticField_msg()
        mag_msg.header.stamp = rospy.get_rostime()
        mag_msg.header.stamp.secs = int(data_collection_time)
        mag_msg.header.stamp.nsecs = int((data_collection_time - imu_msg.header.stamp.secs) * 1000000000)
        mag_msg.magnetic_field.x = float(match.group('mag_x')) - 150.75
        mag_msg.magnetic_field.y = float(match.group('mag_y')) - 7.205
        mag_msg.magnetic_field.z = float(match.group('mag_z')) - -106.045

        set_covariance(mag_msg.magnetic_field_covariance, 1.0)

        imu_pub.publish(imu_msg)
        mag_pub.publish(mag_msg)
    except ValueError as e:
        pass  # there was probably some corrupted data from the serial connection


def map_to(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


if __name__ == "__main__":
    imu_pub = rospy.Publisher('imu/data_raw', Imu_msg, queue_size = 2)
    mag_pub = rospy.Publisher('imu/mag', MagneticField_msg, queue_size = 2)
    gps_pub = rospy.Publisher('gps/nmea_sentence', NmeaSentence_msg, queue_size = 2)

    rospy.init_node('IMU_reader_node')
    main(imu_pub, mag_pub)
